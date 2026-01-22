import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
class Launchpad(Forge):
    """The Launchpad hosting service."""
    supports_merge_proposal_labels = False
    supports_merge_proposal_title = False
    supports_merge_proposal_commit_message = True
    supports_allow_collaboration = False
    merge_proposal_description_format = 'plain'

    def __init__(self, service_root):
        self._api_base_url = service_root
        self._launchpad = None

    @property
    def name(self):
        if self._api_base_url == lp_uris.LPNET_SERVICE_ROOT:
            return 'Launchpad'
        return 'Launchpad at %s' % self.base_url

    @property
    def launchpad(self):
        if self._launchpad is None:
            self._launchpad = lp_api.connect_launchpad(self._api_base_url, version='devel')
        return self._launchpad

    @property
    def base_url(self):
        return lp_uris.web_root_for_service_root(self._api_base_url)

    def __repr__(self):
        return 'Launchpad(service_root=%s)' % self._api_base_url

    def get_current_user(self):
        return self.launchpad.me.name

    def get_user_url(self, username):
        return self.launchpad.people[username].web_link

    def hosts(self, branch):
        return plausible_launchpad_url(branch.user_url)

    @classmethod
    def probe_from_hostname(cls, hostname, possible_transports=None):
        if re.match(hostname, '(bazaar|git).*\\.launchpad\\.net'):
            return Launchpad(lp_uris.LPNET_SERVICE_ROOT)
        raise UnsupportedForge(hostname)

    @classmethod
    def probe_from_url(cls, url, possible_transports=None):
        if plausible_launchpad_url(url):
            return Launchpad(lp_uris.LPNET_SERVICE_ROOT)
        raise UnsupportedForge(url)

    def _get_lp_git_ref_from_branch(self, branch):
        url, params = urlutils.split_segment_parameters(branch.user_url)
        scheme, user, password, host, port, path = urlutils.parse_url(url)
        repo_lp = self.launchpad.git_repositories.getByPath(path=path.strip('/'))
        try:
            ref_path = params['ref']
        except KeyError:
            branch_name = params.get('branch', branch.name)
            if branch_name:
                ref_path = 'refs/heads/%s' % branch_name
            else:
                ref_path = repo_lp.default_branch
        ref_lp = repo_lp.getRefByPath(path=ref_path)
        return (repo_lp, ref_lp)

    def _get_lp_bzr_branch_from_branch(self, branch):
        return self.launchpad.branches.getByUrl(url=urlutils.unescape(branch.user_url))

    def _get_derived_git_path(self, base_path, owner, project):
        base_repo = self.launchpad.git_repositories.getByPath(path=base_path)
        if project is None:
            project = urlutils.parse_url(base_repo.git_ssh_url)[-1].strip('/')
        if project.startswith('~'):
            project = '/'.join(base_path.split('/')[1:])
        return '~{}/{}'.format(owner, project)

    def _publish_git(self, local_branch, base_path, name, owner, project=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
        if tag_selector is None:
            tag_selector = lambda t: False
        to_path = self._get_derived_git_path(base_path, owner, project)
        to_transport = get_transport(GIT_SCHEME_MAP['ssh'] + to_path)
        try:
            dir_to = controldir.ControlDir.open_from_transport(to_transport)
        except errors.NotBranchError:
            dir_to = None
        if dir_to is None:
            try:
                br_to = local_branch.create_clone_on_transport(to_transport, revision_id=revision_id, name=name, tag_selector=tag_selector)
            except errors.NoRoundtrippingSupport:
                br_to = local_branch.create_clone_on_transport(to_transport, revision_id=revision_id, name=name, lossy=True, tag_selector=tag_selector)
        else:
            try:
                dir_to = dir_to.push_branch(local_branch, revision_id, overwrite=overwrite, name=name, tag_selector=tag_selector)
            except errors.NoRoundtrippingSupport:
                if not allow_lossy:
                    raise
                dir_to = dir_to.push_branch(local_branch, revision_id, overwrite=overwrite, name=name, lossy=True, tag_selector=tag_selector)
            br_to = dir_to.target_branch
        return (br_to, 'https://git.launchpad.net/{}/+ref/{}'.format(to_path, name))

    def _get_derived_bzr_path(self, base_branch, name, owner, project):
        if project is None:
            base_branch_lp = self._get_lp_bzr_branch_from_branch(base_branch)
            project = '/'.join(base_branch_lp.unique_name.split('/')[1:-1])
        return '~{}/{}/{}'.format(owner, project, name)

    def get_push_url(self, branch):
        vcs, user, password, path, params = self._split_url(branch.user_url)
        if vcs == 'bzr':
            branch_lp = self._get_lp_bzr_branch_from_branch(branch)
            return branch_lp.bzr_identity
        elif vcs == 'git':
            return urlutils.join_segment_parameters(GIT_SCHEME_MAP['ssh'] + path, params)
        else:
            raise AssertionError

    def _publish_bzr(self, local_branch, base_branch, name, owner, project=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
        to_path = self._get_derived_bzr_path(base_branch, name, owner, project)
        to_transport = get_transport(BZR_SCHEME_MAP['ssh'] + to_path)
        try:
            dir_to = controldir.ControlDir.open_from_transport(to_transport)
        except errors.NotBranchError:
            dir_to = None
        if dir_to is None:
            br_to = local_branch.create_clone_on_transport(to_transport, revision_id=revision_id, tag_selector=tag_selector)
        else:
            br_to = dir_to.push_branch(local_branch, revision_id, overwrite=overwrite, tag_selector=tag_selector).target_branch
        return (br_to, 'https://code.launchpad.net/' + to_path)

    def _split_url(self, url):
        url, params = urlutils.split_segment_parameters(url)
        scheme, user, password, host, port, path = urlutils.parse_url(url)
        path = path.strip('/')
        if host.startswith('bazaar.'):
            vcs = 'bzr'
        elif host.startswith('git.'):
            vcs = 'git'
        else:
            raise ValueError('unknown host %s' % host)
        return (vcs, user, password, path, params)

    def publish_derived(self, local_branch, base_branch, name, project=None, owner=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
        """Publish a branch to the site, derived from base_branch.

        :param base_branch: branch to derive the new branch from
        :param new_branch: branch to publish
        :param name: Name of the new branch on the remote host
        :param project: Optional project name
        :param owner: Optional owner
        :return: resulting branch
        """
        if owner is None:
            owner = self.launchpad.me.name
        base_vcs, base_user, base_password, base_path, base_params = self._split_url(base_branch.user_url)
        if base_vcs == 'bzr':
            return self._publish_bzr(local_branch, base_branch, name, project=project, owner=owner, revision_id=revision_id, overwrite=overwrite, allow_lossy=allow_lossy, tag_selector=tag_selector)
        elif base_vcs == 'git':
            return self._publish_git(local_branch, base_path, name, project=project, owner=owner, revision_id=revision_id, overwrite=overwrite, allow_lossy=allow_lossy, tag_selector=tag_selector)
        else:
            raise AssertionError('not a valid Launchpad URL')

    def get_derived_branch(self, base_branch, name, project=None, owner=None, preferred_schemes=None):
        if preferred_schemes is None:
            preferred_schemes = DEFAULT_PREFERRED_SCHEMES
        if owner is None:
            owner = self.launchpad.me.name
        base_vcs, base_user, base_password, base_path, base_params = self._split_url(base_branch.user_url)
        if base_vcs == 'bzr':
            to_path = self._get_derived_bzr_path(base_branch, name, owner, project)
            for scheme in preferred_schemes:
                try:
                    prefix = BZR_SCHEME_MAP[scheme]
                except KeyError:
                    continue
                return _mod_branch.Branch.open(prefix + to_path)
            raise AssertionError('no supported schemes: %r' % preferred_schemes)
        elif base_vcs == 'git':
            to_path = self._get_derived_git_path(base_path.strip('/'), owner, project)
            for scheme in preferred_schemes:
                try:
                    prefix = GIT_SCHEME_MAP[scheme]
                except KeyError:
                    continue
                to_url = urlutils.join_segment_parameters(prefix + to_path, {'branch': name})
                return _mod_branch.Branch.open(to_url)
            raise AssertionError('no supported schemes: %r' % preferred_schemes)
        else:
            raise AssertionError('not a valid Launchpad URL')

    def iter_proposals(self, source_branch, target_branch, status='open'):
        base_vcs, base_user, base_password, base_path, base_params = self._split_url(target_branch.user_url)
        statuses = status_to_lp_mp_statuses(status)
        if base_vcs == 'bzr':
            target_branch_lp = self.launchpad.branches.getByUrl(url=target_branch.user_url)
            source_branch_lp = self.launchpad.branches.getByUrl(url=source_branch.user_url)
            for mp in target_branch_lp.getMergeProposals(status=statuses):
                if mp.source_branch_link != source_branch_lp.self_link:
                    continue
                yield LaunchpadMergeProposal(mp)
        elif base_vcs == 'git':
            source_repo_lp, source_branch_lp = self._get_lp_git_ref_from_branch(source_branch)
            target_repo_lp, target_branch_lp = self._get_lp_git_ref_from_branch(target_branch)
            for mp in target_branch_lp.getMergeProposals(status=statuses):
                if target_branch_lp.path != mp.target_git_path or target_repo_lp != mp.target_git_repository or source_branch_lp.path != mp.source_git_path or (source_repo_lp != mp.source_git_repository):
                    continue
                yield LaunchpadMergeProposal(mp)
        else:
            raise AssertionError('not a valid Launchpad URL')

    def get_proposer(self, source_branch, target_branch):
        base_vcs, base_user, base_password, base_path, base_params = self._split_url(target_branch.user_url)
        if base_vcs == 'bzr':
            return LaunchpadBazaarMergeProposalBuilder(self, source_branch, target_branch)
        elif base_vcs == 'git':
            return LaunchpadGitMergeProposalBuilder(self, source_branch, target_branch)
        else:
            raise AssertionError('not a valid Launchpad URL')

    @classmethod
    def iter_instances(cls):
        credential_store = lp_api.get_credential_store()
        for service_root in set(lp_uris.service_roots.values()):
            auth_engine = lp_api.get_auth_engine(service_root)
            creds = credential_store.load(auth_engine.unique_consumer_id)
            if creds is not None:
                yield cls(service_root)

    def iter_my_proposals(self, status='open', author=None):
        statuses = status_to_lp_mp_statuses(status)
        if author is None:
            author_obj = self.launchpad.me
        else:
            author_obj = self._getPerson(author)
        for mp in author_obj.getMergeProposals(status=statuses):
            yield LaunchpadMergeProposal(mp)

    def iter_my_forks(self, owner=None):
        return iter([])

    def _getPerson(self, person):
        if '@' in person:
            return self.launchpad.people.getByEmail(email=person)
        else:
            return self.launchpad.people[person]

    def get_web_url(self, branch):
        vcs, user, password, path, params = self._split_url(branch.user_url)
        if vcs == 'bzr':
            branch_lp = self._get_lp_bzr_branch_from_branch(branch)
            return branch_lp.web_link
        elif vcs == 'git':
            repo_lp, ref_lp = self._get_lp_git_ref_from_branch(branch)
            return ref_lp.web_link
        else:
            raise AssertionError

    def get_proposal_by_url(self, url):
        scheme, user, password, host, port, path = urlutils.parse_url(url)
        LAUNCHPAD_CODE_DOMAINS = ['code.%s' % domain for domain in lp_uris.LAUNCHPAD_DOMAINS.values()]
        if host not in LAUNCHPAD_CODE_DOMAINS:
            raise UnsupportedForge(url)
        api_url = str(self.launchpad._root_uri) + path
        mp = self.launchpad.load(api_url)
        return LaunchpadMergeProposal(mp)

    def create_project(self, path, summary=None):
        self.launchpad.projects.new_project(display_name=path, name=path, summary=summary, title=path)