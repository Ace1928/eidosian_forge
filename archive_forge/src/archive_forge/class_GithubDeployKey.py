from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from re import findall
class GithubDeployKey(object):

    def __init__(self, module):
        self.module = module
        self.github_url = self.module.params['github_url']
        self.name = module.params['name']
        self.key = module.params['key']
        self.state = module.params['state']
        self.read_only = module.params.get('read_only', True)
        self.force = module.params.get('force', False)
        self.username = module.params.get('username', None)
        self.password = module.params.get('password', None)
        self.token = module.params.get('token', None)
        self.otp = module.params.get('otp', None)

    @property
    def url(self):
        owner = self.module.params['owner']
        repo = self.module.params['repo']
        return '{0}/repos/{1}/{2}/keys'.format(self.github_url, owner, repo)

    @property
    def headers(self):
        if self.username is not None and self.password is not None:
            self.module.params['url_username'] = self.username
            self.module.params['url_password'] = self.password
            self.module.params['force_basic_auth'] = True
            if self.otp is not None:
                return {'X-GitHub-OTP': self.otp}
        elif self.token is not None:
            return {'Authorization': 'token {0}'.format(self.token)}
        else:
            return None

    def paginate(self, url):
        while url:
            resp, info = fetch_url(self.module, url, headers=self.headers, method='GET')
            if info['status'] == 200:
                yield self.module.from_json(resp.read())
                links = {}
                for x, y in findall('<([^>]+)>;\\s*rel="(\\w+)"', info.get('link', '')):
                    links[y] = x
                url = links.get('next')
            else:
                self.handle_error(method='GET', info=info)

    def get_existing_key(self):
        for keys in self.paginate(self.url):
            if keys:
                for i in keys:
                    existing_key_id = str(i['id'])
                    if i['key'].split() == self.key.split()[:2]:
                        return existing_key_id
                    elif i['title'] == self.name and self.force:
                        return existing_key_id
            else:
                return None

    def add_new_key(self):
        request_body = {'title': self.name, 'key': self.key, 'read_only': self.read_only}
        resp, info = fetch_url(self.module, self.url, data=self.module.jsonify(request_body), headers=self.headers, method='POST', timeout=30)
        status_code = info['status']
        if status_code == 201:
            response_body = self.module.from_json(resp.read())
            key_id = response_body['id']
            self.module.exit_json(changed=True, msg='Deploy key successfully added', id=key_id)
        elif status_code == 422:
            self.module.exit_json(changed=False, msg='Deploy key already exists')
        else:
            self.handle_error(method='POST', info=info)

    def remove_existing_key(self, key_id):
        resp, info = fetch_url(self.module, '{0}/{1}'.format(self.url, key_id), headers=self.headers, method='DELETE')
        status_code = info['status']
        if status_code == 204:
            if self.state == 'absent':
                self.module.exit_json(changed=True, msg='Deploy key successfully deleted', id=key_id)
        else:
            self.handle_error(method='DELETE', info=info, key_id=key_id)

    def handle_error(self, method, info, key_id=None):
        status_code = info['status']
        body = info.get('body')
        if body:
            err = self.module.from_json(body)['message']
        if status_code == 401:
            self.module.fail_json(msg='Failed to connect to {0} due to invalid credentials'.format(self.github_url), http_status_code=status_code, error=err)
        elif status_code == 404:
            self.module.fail_json(msg='GitHub repository does not exist', http_status_code=status_code, error=err)
        elif method == 'GET':
            self.module.fail_json(msg='Failed to retrieve existing deploy keys', http_status_code=status_code, error=err)
        elif method == 'POST':
            self.module.fail_json(msg='Failed to add deploy key', http_status_code=status_code, error=err)
        elif method == 'DELETE':
            self.module.fail_json(msg='Failed to delete existing deploy key', id=key_id, http_status_code=status_code, error=err)