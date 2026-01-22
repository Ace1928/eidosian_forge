import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
class GitLab(Forge):
    """GitLab forge implementation."""
    supports_merge_proposal_labels = True
    supports_merge_proposal_title = True
    supports_merge_proposal_commit_message = False
    supports_allow_collaboration = True
    merge_proposal_description_format = 'markdown'

    def __repr__(self):
        return '<GitLab(%r)>' % self.base_url

    @property
    def base_url(self):
        return self.transport.base

    @property
    def base_hostname(self):
        return urlutils.parse_url(self.base_url)[3]

    def _find_correct_project_name(self, path):
        try:
            resp = self.transport.request('GET', urlutils.join(self.base_url, path), headers=self.headers)
        except errors.RedirectRequested as e:
            return urlutils.parse_url(e.target)[-1].strip('/')
        if resp.status != 200:
            _unexpected_status(path, resp)
        return None

    def _api_request(self, method, path, fields=None, body=None):
        return self.transport.request(method, urlutils.join(self.base_url, 'api', 'v4', path), headers=self.headers, fields=fields, body=body)

    def __init__(self, transport, private_token):
        self.transport = transport
        self.headers = {'Private-Token': private_token}
        self._current_user = None

    def _get_user(self, username):
        path = 'users/%s' % urlutils.quote(str(username), '')
        response = self._api_request('GET', path)
        if response.status == 404:
            raise KeyError('no such user %s' % username)
        if response.status == 200:
            return json.loads(response.data)
        _unexpected_status(path, response)

    def _get_user_by_email(self, email):
        path = 'users?search=%s' % urlutils.quote(str(email), '')
        response = self._api_request('GET', path)
        if response.status == 404:
            raise KeyError('no such user %s' % email)
        if response.status == 200:
            ret = json.loads(response.data)
            if len(ret) != 1:
                raise ValueError('unexpected number of results; %r' % ret)
            return ret[0]
        _unexpected_status(path, response)

    def _get_project(self, project_name, _redirect_checked=False):
        path = 'projects/%s' % urlutils.quote(str(project_name), '')
        response = self._api_request('GET', path)
        if response.status == 404:
            if not _redirect_checked:
                project_name = self._find_correct_project_name(project_name)
                if project_name is not None:
                    return self._get_project(project_name, _redirect_checked=True)
            raise NoSuchProject(project_name)
        if response.status == 200:
            return json.loads(response.data)
        _unexpected_status(path, response)

    def _get_namespace(self, namespace):
        path = 'namespaces/' + urlutils.quote(str(namespace), '')
        response = self._api_request('GET', path)
        if response.status == 200:
            return json.loads(response.data)
        if response.status == 404:
            return None
        _unexpected_status(path, response)

    def create_project(self, project_name, summary=None):
        if project_name.endswith('.git'):
            project_name = project_name[:-4]
        if '/' in project_name:
            namespace_path, path = project_name.lstrip('/').rsplit('/', 1)
        else:
            namespace_path = ''
            path = project_name
        namespace = self._get_namespace(namespace_path)
        if namespace is None:
            raise Exception('namespace %s does not exist' % namespace_path)
        fields = {'path': path, 'namespace_id': namespace['id']}
        if summary is not None:
            fields['description'] = summary
        response = self._api_request('POST', 'projects', fields=fields)
        if response.status == 400:
            ret = json.loads(response.data)
            if ret.get('message', {}).get('path') == ['has already been taken']:
                raise errors.AlreadyControlDirError(project_name)
            raise
        if response.status == 403:
            raise errors.PermissionDenied(response.text)
        if response.status not in (200, 201):
            _unexpected_status('projects', response)
        project = json.loads(response.data)
        return project

    def fork_project(self, project_name, timeout=50, interval=5, owner=None):
        path = 'projects/%s/fork' % urlutils.quote(str(project_name), '')
        fields = {}
        if owner is not None:
            fields['namespace'] = owner
        response = self._api_request('POST', path, fields=fields)
        if response.status == 404:
            raise ForkingDisabled(project_name)
        if response.status == 409:
            resp = json.loads(response.data)
            raise GitLabConflict(resp.get('message'))
        if response.status not in (200, 201):
            _unexpected_status(path, response)
        project = json.loads(response.data)
        deadline = time.time() + timeout
        while project['import_status'] not in ('finished', 'none'):
            mutter('import status is %s', project['import_status'])
            if time.time() > deadline:
                raise ProjectCreationTimeout(project['path_with_namespace'], timeout)
            time.sleep(interval)
            project = self._get_project(project['path_with_namespace'])
        return project

    def _handle_merge_request_conflict(self, message, source_url, target_project):
        m = re.fullmatch('Another open merge request already exists for this source branch: \\!([0-9]+)', message[0])
        if m:
            merge_id = int(m.group(1))
            mr = self._get_merge_request(target_project, merge_id)
            raise MergeProposalExists(source_url, GitLabMergeProposal(self, mr))
        raise MergeRequestConflict(message)

    def get_current_user(self):
        if not self._current_user:
            self._retrieve_user()
        return self._current_user['username']

    def get_user_url(self, username):
        return urlutils.join(self.base_url, username)

    def _list_paged(self, path, parameters=None, per_page=None):
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters.items())
        if per_page:
            parameters['per_page'] = str(per_page)
        page = '1'
        while page:
            parameters['page'] = page
            response = self._api_request('GET', path + '?' + '&'.join(['%s=%s' % item for item in parameters.items()]))
            if response.status == 403:
                raise errors.PermissionDenied(response.text)
            if response.status != 200:
                _unexpected_status(path, response)
            page = response.getheader('X-Next-Page')
            yield from json.loads(response.data)

    def _list_merge_requests(self, author=None, project=None, state=None):
        if project is not None:
            path = 'projects/%s/merge_requests' % urlutils.quote(str(project), '')
        else:
            path = 'merge_requests'
        parameters = {}
        if state:
            parameters['state'] = state
        if author:
            parameters['author_username'] = urlutils.quote(author, '')
        return self._list_paged(path, parameters, per_page=DEFAULT_PAGE_SIZE)

    def _get_merge_request(self, project, merge_id):
        path = 'projects/%s/merge_requests/%d' % (urlutils.quote(str(project), ''), merge_id)
        response = self._api_request('GET', path)
        if response.status == 403:
            raise errors.PermissionDenied(response.text)
        if response.status != 200:
            _unexpected_status(path, response)
        return json.loads(response.data)

    def _list_projects(self, owner):
        path = 'users/%s/projects' % urlutils.quote(str(owner), '')
        parameters = {}
        return self._list_paged(path, parameters, per_page=DEFAULT_PAGE_SIZE)

    def _update_merge_request(self, project_id, iid, mr):
        path = 'projects/{}/merge_requests/{}'.format(urlutils.quote(str(project_id), ''), iid)
        response = self._api_request('PUT', path, fields=mr)
        if response.status == 200:
            return json.loads(response.data)
        if response.status == 409:
            raise GitLabConflict(json.loads(response.data).get('message'))
        if response.status == 403:
            raise errors.PermissionDenied(response.text)
        _unexpected_status(path, response)

    def _merge_mr(self, project_id, iid, kwargs):
        path = 'projects/{}/merge_requests/{}/merge'.format(urlutils.quote(str(project_id), ''), iid)
        response = self._api_request('PUT', path, fields=kwargs)
        if response.status == 200:
            return json.loads(response.data)
        if response.status == 403:
            raise errors.PermissionDenied(response.text)
        _unexpected_status(path, response)

    def _post_merge_request_note(self, project_id, iid, kwargs):
        path = 'projects/{}/merge_requests/{}/notes'.format(urlutils.quote(str(project_id), ''), iid)
        response = self._api_request('POST', path, fields=kwargs)
        if response.status == 201:
            json.loads(response.data)
            return
        if response.status == 403:
            raise errors.PermissionDenied(response.text)
        _unexpected_status(path, response)

    def _create_mergerequest(self, title, source_project_id, target_project_id, source_branch_name, target_branch_name, description, labels=None, allow_collaboration=False):
        path = 'projects/%s/merge_requests' % source_project_id
        fields = {'title': title, 'source_branch': source_branch_name, 'target_branch': target_branch_name, 'target_project_id': target_project_id, 'description': description, 'allow_collaboration': allow_collaboration}
        if labels:
            fields['labels'] = labels
        response = self._api_request('POST', path, fields=fields)
        if response.status == 400:
            data = json.loads(response.data)
            raise GitLabError(data.get('message'), response)
        if response.status == 403:
            raise errors.PermissionDenied(response.text)
        if response.status == 409:
            raise GitLabConflict(json.loads(response.data).get('message'))
        if response.status == 422:
            data = json.loads(response.data)
            raise GitLabUnprocessable(data.get('error') or data.get('message'), data)
        if response.status != 201:
            _unexpected_status(path, response)
        return json.loads(response.data)

    def get_push_url(self, branch):
        host, project_name, branch_name = parse_gitlab_branch_url(branch)
        project = self._get_project(project_name)
        return gitlab_url_to_bzr_url(project['ssh_url_to_repo'], branch_name)

    def get_web_url(self, branch):
        host, project_name, branch_name = parse_gitlab_branch_url(branch)
        project = self._get_project(project_name)
        if branch_name:
            return project['web_url'] + '/-/tree/' + branch_name
        else:
            return project['web_url']

    def publish_derived(self, local_branch, base_branch, name, project=None, owner=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
        if tag_selector is None:
            tag_selector = lambda t: False
        host, base_project_name, base_branch_name = parse_gitlab_branch_url(base_branch)
        if owner is None:
            owner = base_branch.get_config_stack().get('fork-namespace')
        if owner is None:
            owner = self.get_current_user()
        base_project = self._get_project(base_project_name)
        if project is None:
            project = base_project['path']
        try:
            target_project = self._get_project('{}/{}'.format(owner, project))
        except NoSuchProject:
            target_project = self.fork_project(base_project['path_with_namespace'], owner=owner)
        remote_repo_url = git_url_to_bzr_url(target_project['ssh_url_to_repo'])
        remote_dir = controldir.ControlDir.open(remote_repo_url)
        try:
            push_result = remote_dir.push_branch(local_branch, revision_id=revision_id, overwrite=overwrite, name=name, tag_selector=tag_selector)
        except errors.NoRoundtrippingSupport:
            if not allow_lossy:
                raise
            push_result = remote_dir.push_branch(local_branch, revision_id=revision_id, overwrite=overwrite, name=name, lossy=True, tag_selector=tag_selector)
        public_url = gitlab_url_to_bzr_url(target_project['http_url_to_repo'], name)
        return (push_result.target_branch, public_url)

    def get_derived_branch(self, base_branch, name, project=None, owner=None, preferred_schemes=None):
        host, base_project, base_branch_name = parse_gitlab_branch_url(base_branch)
        if owner is None:
            owner = self.get_current_user()
        if project is None:
            project = self._get_project(base_project)['path']
        try:
            target_project = self._get_project('{}/{}'.format(owner, project))
        except NoSuchProject:
            raise errors.NotBranchError('{}/{}/{}'.format(self.base_url, owner, project))
        if preferred_schemes is None:
            preferred_schemes = ['git+ssh']
        for scheme in preferred_schemes:
            if scheme == 'git+ssh':
                gitlab_url = target_project['ssh_url_to_repo']
                break
            elif scheme == 'https':
                gitlab_url = target_project['http_url_to_repo']
                break
        else:
            raise AssertionError
        return _mod_branch.Branch.open(gitlab_url_to_bzr_url(gitlab_url, name), possible_transports=[base_branch.user_transport])

    def get_proposer(self, source_branch, target_branch):
        return GitlabMergeProposalBuilder(self, source_branch, target_branch)

    def iter_proposals(self, source_branch, target_branch, status):
        source_host, source_project_name, source_branch_name = parse_gitlab_branch_url(source_branch)
        target_host, target_project_name, target_branch_name = parse_gitlab_branch_url(target_branch)
        if source_host != target_host:
            raise DifferentGitLabInstances(source_host, target_host)
        source_project = self._get_project(source_project_name)
        target_project = self._get_project(target_project_name)
        state = mp_status_to_status(status)
        for mr in self._list_merge_requests(project=target_project['id'], state=state):
            if mr['source_project_id'] != source_project['id'] or mr['source_branch'] != source_branch_name or mr['target_project_id'] != target_project['id'] or (mr['target_branch'] != target_branch_name):
                continue
            yield GitLabMergeProposal(self, mr)

    def hosts(self, branch):
        try:
            host, project, branch_name = parse_gitlab_branch_url(branch)
        except NotGitLabUrl:
            return False
        return self.base_hostname == host

    def _retrieve_user(self):
        if self._current_user:
            return
        try:
            response = self._api_request('GET', 'user')
        except errors.UnexpectedHttpStatus as e:
            if e.code == 401:
                raise GitLabLoginMissing(self.base_url)
            raise
        if response.status == 200:
            self._current_user = json.loads(response.data)
            return
        if response.status == 401:
            if json.loads(response.data) == {'message': '401 Unauthorized'}:
                raise GitLabLoginMissing(self.base_url)
            else:
                raise GitlabLoginError(response.text)
        raise UnsupportedForge(self.base_url)

    @classmethod
    def probe_from_hostname(cls, hostname, possible_transports=None):
        base_url = 'https://%s' % hostname
        credentials = get_credentials_by_url(base_url)
        if credentials is not None:
            transport = get_transport(base_url, possible_transports=possible_transports)
            instance = cls(transport, credentials.get('private_token'))
            instance._retrieve_user()
            return instance
        raise UnsupportedForge(hostname)

    @classmethod
    def probe_from_url(cls, url, possible_transports=None):
        try:
            host, project = parse_gitlab_url(url)
        except NotGitLabUrl:
            raise UnsupportedForge(url)
        transport = get_transport('https://%s' % host, possible_transports=possible_transports)
        credentials = get_credentials_by_url(transport.base)
        if credentials is not None:
            instance = cls(transport, credentials.get('private_token'))
            instance._retrieve_user()
            return instance
        try:
            resp = transport.request('GET', 'https://{}/api/v4/projects/{}'.format(host, urlutils.quote(str(project), '')))
        except errors.UnexpectedHttpStatus as e:
            raise UnsupportedForge(url)
        except errors.RedirectRequested:
            raise UnsupportedForge(url)
        else:
            if not resp.getheader('X-Gitlab-Feature-Category'):
                raise UnsupportedForge(url)
            if resp.status in (200, 401):
                raise GitLabLoginMissing('https://%s/' % host)
            raise UnsupportedForge(url)

    @classmethod
    def iter_instances(cls):
        for name, credentials in iter_tokens():
            if 'url' not in credentials:
                continue
            yield cls(get_transport(credentials['url']), private_token=credentials.get('private_token'))

    def iter_my_proposals(self, status='open', author=None):
        if author is None:
            author = self.get_current_user()
        state = mp_status_to_status(status)
        for mp in self._list_merge_requests(author=author, state=state):
            yield GitLabMergeProposal(self, mp)

    def iter_my_forks(self, owner: Optional[str]=None):
        if owner is None:
            owner = self.get_current_user()
        for project in self._list_projects(owner=owner):
            base_project = project.get('forked_from_project')
            if not base_project:
                continue
            yield project['path_with_namespace']

    def get_proposal_by_url(self, url: str) -> GitLabMergeProposal:
        try:
            host, project, merge_id = parse_gitlab_merge_request_url(url)
        except NotGitLabUrl:
            raise UnsupportedForge(url)
        except NotMergeRequestUrl as e:
            if self.base_hostname == e.host:
                raise
            else:
                raise UnsupportedForge(url)
        if self.base_hostname != host:
            raise UnsupportedForge(url)
        project = self._get_project(project)
        mr = self._get_merge_request(project['path_with_namespace'], merge_id)
        return GitLabMergeProposal(self, mr)

    def delete_project(self, project):
        path = 'projects/%s' % urlutils.quote(str(project), '')
        response = self._api_request('DELETE', path)
        if response.status == 404:
            raise NoSuchProject(project)
        if response.status != 202:
            _unexpected_status(path, response)