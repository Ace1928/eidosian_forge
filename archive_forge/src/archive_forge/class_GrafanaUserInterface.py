from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
class GrafanaUserInterface(object):

    def __init__(self, module):
        self._module = module
        self.headers = {'Content-Type': 'application/json'}
        self.headers['Authorization'] = basic_auth_header(module.params['url_username'], module.params['url_password'])
        self.grafana_url = base.clean_url(module.params.get('url'))

    def _send_request(self, url, data=None, headers=None, method='GET'):
        if data is not None:
            data = json.dumps(data, sort_keys=True)
        if not headers:
            headers = []
        full_url = '{grafana_url}{path}'.format(grafana_url=self.grafana_url, path=url)
        resp, info = fetch_url(self._module, full_url, data=data, headers=headers, method=method)
        status_code = info['status']
        if status_code == 404:
            return None
        elif status_code == 401:
            self._module.fail_json(failed=True, msg="Unauthorized to perform action '%s' on '%s' header: %s" % (method, full_url, self.headers))
        elif status_code == 403:
            self._module.fail_json(failed=True, msg='Permission Denied')
        elif status_code == 200:
            return self._module.from_json(resp.read())
        self._module.fail_json(failed=True, msg='Grafana Users API answered with HTTP %d' % status_code, body=self._module.from_json(resp.read()))

    def create_user(self, name, email, login, password):
        if not password:
            self._module.fail_json(failed=True, msg='missing required arguments: password')
        url = '/api/admin/users'
        user = dict(name=name, email=email, login=login, password=password)
        self._send_request(url, data=user, headers=self.headers, method='POST')
        return self.get_user_from_login(login)

    def get_user_from_login(self, login):
        url = '/api/users/lookup?loginOrEmail={login}'.format(login=quote(login))
        return self._send_request(url, headers=self.headers, method='GET')

    def update_user(self, user_id, email, name, login):
        url = '/api/users/{user_id}'.format(user_id=user_id)
        user = dict(email=email, name=name, login=login)
        self._send_request(url, data=user, headers=self.headers, method='PUT')
        return self.get_user_from_login(login)

    def update_user_permissions(self, user_id, is_admin):
        url = '/api/admin/users/{user_id}/permissions'.format(user_id=user_id)
        permissions = dict(isGrafanaAdmin=is_admin)
        return self._send_request(url, data=permissions, headers=self.headers, method='PUT')

    def delete_user(self, user_id):
        url = '/api/admin/users/{user_id}'.format(user_id=user_id)
        return self._send_request(url, headers=self.headers, method='DELETE')