from __future__ import absolute_import, division, print_function
import base64
import binascii
import json
import mimetypes
import os
import random
import string
import traceback
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper, cause_changes
from ansible.module_utils.six.moves.urllib.request import pathname2url
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.urls import fetch_url
class JIRA(StateModuleHelper):
    module = dict(argument_spec=dict(attachment=dict(type='dict', options=dict(content=dict(type='str'), filename=dict(type='path', required=True), mimetype=dict(type='str'))), uri=dict(type='str', required=True), operation=dict(type='str', choices=['attach', 'create', 'comment', 'edit', 'update', 'fetch', 'transition', 'link', 'search', 'worklog'], aliases=['command'], required=True), username=dict(type='str'), password=dict(type='str', no_log=True), token=dict(type='str', no_log=True), project=dict(type='str'), summary=dict(type='str'), description=dict(type='str'), issuetype=dict(type='str'), issue=dict(type='str', aliases=['ticket']), comment=dict(type='str'), comment_visibility=dict(type='dict', options=dict(type=dict(type='str', choices=['group', 'role'], required=True), value=dict(type='str', required=True))), status=dict(type='str'), assignee=dict(type='str'), fields=dict(default={}, type='dict'), linktype=dict(type='str'), inwardissue=dict(type='str'), outwardissue=dict(type='str'), jql=dict(type='str'), maxresults=dict(type='int'), timeout=dict(type='float', default=10), validate_certs=dict(default=True, type='bool'), account_id=dict(type='str')), mutually_exclusive=[['username', 'token'], ['password', 'token'], ['assignee', 'account_id']], required_together=[['username', 'password']], required_one_of=[['username', 'token']], required_if=(('operation', 'attach', ['issue', 'attachment']), ('operation', 'create', ['project', 'issuetype', 'summary']), ('operation', 'comment', ['issue', 'comment']), ('operation', 'workflow', ['issue', 'comment']), ('operation', 'fetch', ['issue']), ('operation', 'transition', ['issue', 'status']), ('operation', 'link', ['linktype', 'inwardissue', 'outwardissue']), ('operation', 'search', ['jql'])), supports_check_mode=False)
    state_param = 'operation'

    def __init_module__(self):
        if self.vars.fields is None:
            self.vars.fields = {}
        if self.vars.assignee:
            self.vars.fields['assignee'] = {'name': self.vars.assignee}
        if self.vars.account_id:
            self.vars.fields['assignee'] = {'accountId': self.vars.account_id}
        self.vars.uri = self.vars.uri.strip('/')
        self.vars.set('restbase', self.vars.uri + '/rest/api/2')

    @cause_changes(on_success=True)
    def operation_create(self):
        createfields = {'project': {'key': self.vars.project}, 'summary': self.vars.summary, 'issuetype': {'name': self.vars.issuetype}}
        if self.vars.description:
            createfields['description'] = self.vars.description
        if self.vars.fields:
            createfields.update(self.vars.fields)
        data = {'fields': createfields}
        url = self.vars.restbase + '/issue/'
        self.vars.meta = self.post(url, data)

    @cause_changes(on_success=True)
    def operation_comment(self):
        data = {'body': self.vars.comment}
        if self.vars.comment_visibility is not None:
            data['visibility'] = self.vars.comment_visibility
        if self.vars.fields:
            data.update(self.vars.fields)
        url = self.vars.restbase + '/issue/' + self.vars.issue + '/comment'
        self.vars.meta = self.post(url, data)

    @cause_changes(on_success=True)
    def operation_worklog(self):
        data = {'comment': self.vars.comment}
        if self.vars.comment_visibility is not None:
            data['visibility'] = self.vars.comment_visibility
        if self.vars.fields:
            data.update(self.vars.fields)
        url = self.vars.restbase + '/issue/' + self.vars.issue + '/worklog'
        self.vars.meta = self.post(url, data)

    @cause_changes(on_success=True)
    def operation_edit(self):
        data = {'fields': self.vars.fields}
        url = self.vars.restbase + '/issue/' + self.vars.issue
        self.vars.meta = self.put(url, data)

    @cause_changes(on_success=True)
    def operation_update(self):
        data = {'update': self.vars.fields}
        url = self.vars.restbase + '/issue/' + self.vars.issue
        self.vars.meta = self.put(url, data)

    def operation_fetch(self):
        url = self.vars.restbase + '/issue/' + self.vars.issue
        self.vars.meta = self.get(url)

    def operation_search(self):
        url = self.vars.restbase + '/search?jql=' + pathname2url(self.vars.jql)
        if self.vars.fields:
            fields = self.vars.fields.keys()
            url = url + '&fields=' + '&fields='.join([pathname2url(f) for f in fields])
        if self.vars.maxresults:
            url = url + '&maxResults=' + str(self.vars.maxresults)
        self.vars.meta = self.get(url)

    @cause_changes(on_success=True)
    def operation_transition(self):
        turl = self.vars.restbase + '/issue/' + self.vars.issue + '/transitions'
        tmeta = self.get(turl)
        target = self.vars.status
        tid = None
        for t in tmeta['transitions']:
            if t['name'] == target:
                tid = t['id']
                break
        else:
            raise ValueError("Failed find valid transition for '%s'" % target)
        fields = dict(self.vars.fields)
        if self.vars.summary is not None:
            fields.update({'summary': self.vars.summary})
        if self.vars.description is not None:
            fields.update({'description': self.vars.description})
        data = {'transition': {'id': tid}, 'fields': fields}
        if self.vars.comment is not None:
            data.update({'update': {'comment': [{'add': {'body': self.vars.comment}}]}})
        url = self.vars.restbase + '/issue/' + self.vars.issue + '/transitions'
        self.vars.meta = self.post(url, data)

    @cause_changes(on_success=True)
    def operation_link(self):
        data = {'type': {'name': self.vars.linktype}, 'inwardIssue': {'key': self.vars.inwardissue}, 'outwardIssue': {'key': self.vars.outwardissue}}
        url = self.vars.restbase + '/issueLink/'
        self.vars.meta = self.post(url, data)

    @cause_changes(on_success=True)
    def operation_attach(self):
        v = self.vars
        filename = v.attachment.get('filename')
        content = v.attachment.get('content')
        if not any((filename, content)):
            raise ValueError('at least one of filename or content must be provided')
        mime = v.attachment.get('mimetype')
        if not os.path.isfile(filename):
            raise ValueError('The provided filename does not exist: %s' % filename)
        content_type, data = self._prepare_attachment(filename, content, mime)
        url = v.restbase + '/issue/' + v.issue + '/attachments'
        return (True, self.post(url, data, content_type=content_type, additional_headers={'X-Atlassian-Token': 'no-check'}))

    @staticmethod
    def _prepare_attachment(filename, content=None, mime_type=None):

        def escape_quotes(s):
            return s.replace('"', '\\"')
        boundary = ''.join((random.choice(string.digits + string.ascii_letters) for dummy in range(30)))
        name = to_native(os.path.basename(filename))
        if not mime_type:
            try:
                mime_type = mimetypes.guess_type(filename or '', strict=False)[0] or 'application/octet-stream'
            except Exception:
                mime_type = 'application/octet-stream'
        main_type, sep, sub_type = mime_type.partition('/')
        if not content and filename:
            with open(to_bytes(filename, errors='surrogate_or_strict'), 'rb') as f:
                content = f.read()
        else:
            try:
                content = base64.b64decode(content)
            except binascii.Error as e:
                raise Exception('Unable to base64 decode file content: %s' % e)
        lines = ['--{0}'.format(boundary), 'Content-Disposition: form-data; name="file"; filename={0}'.format(escape_quotes(name)), 'Content-Type: {0}'.format('{0}/{1}'.format(main_type, sub_type)), '', to_text(content), '--{0}--'.format(boundary), '']
        return ('multipart/form-data; boundary={0}'.format(boundary), '\r\n'.join(lines))

    def request(self, url, data=None, method=None, content_type='application/json', additional_headers=None):
        if data and content_type == 'application/json':
            data = json.dumps(data)
        headers = {}
        if isinstance(additional_headers, dict):
            headers = additional_headers.copy()
        if self.vars.token is not None:
            headers.update({'Content-Type': content_type, 'Authorization': 'Bearer %s' % self.vars.token})
        else:
            auth = to_text(base64.b64encode(to_bytes('{0}:{1}'.format(self.vars.username, self.vars.password), errors='surrogate_or_strict')))
            headers.update({'Content-Type': content_type, 'Authorization': 'Basic %s' % auth})
        response, info = fetch_url(self.module, url, data=data, method=method, timeout=self.vars.timeout, headers=headers)
        if info['status'] not in (200, 201, 204):
            error = None
            try:
                error = json.loads(info['body'])
            except Exception:
                msg = 'The request "{method} {url}" returned the unexpected status code {status} {msg}\n{body}'.format(status=info['status'], msg=info['msg'], body=info.get('body'), url=url, method=method)
                self.module.fail_json(msg=to_native(msg), exception=traceback.format_exc())
            if error:
                msg = []
                for key in ('errorMessages', 'errors'):
                    if error.get(key):
                        msg.append(to_native(error[key]))
                if msg:
                    self.module.fail_json(msg=', '.join(msg))
                self.module.fail_json(msg=to_native(error))
            self.module.fail_json(msg=to_native(info['body']))
        body = response.read()
        if body:
            return json.loads(to_text(body, errors='surrogate_or_strict'))
        return {}

    def post(self, url, data, content_type='application/json', additional_headers=None):
        return self.request(url, data=data, method='POST', content_type=content_type, additional_headers=additional_headers)

    def put(self, url, data):
        return self.request(url, data=data, method='PUT')

    def get(self, url):
        return self.request(url)