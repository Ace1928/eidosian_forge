from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
import codecs
class NitroAPICaller(object):
    _argument_spec = dict(nsip=dict(fallback=(env_fallback, ['NETSCALER_NSIP'])), nitro_user=dict(fallback=(env_fallback, ['NETSCALER_NITRO_USER'])), nitro_pass=dict(fallback=(env_fallback, ['NETSCALER_NITRO_PASS']), no_log=True), nitro_protocol=dict(choices=['http', 'https'], fallback=(env_fallback, ['NETSCALER_NITRO_PROTOCOL']), default='http'), validate_certs=dict(default=True, type='bool'), nitro_auth_token=dict(type='str', no_log=True), resource=dict(type='str'), name=dict(type='str'), attributes=dict(type='dict'), args=dict(type='dict'), filter=dict(type='dict'), operation=dict(type='str', required=True, choices=['add', 'update', 'get', 'get_by_args', 'get_filtered', 'get_all', 'delete', 'delete_by_args', 'count', 'mas_login', 'save_config', 'action']), expected_nitro_errorcode=dict(type='list', default=[0]), action=dict(type='str'), instance_ip=dict(type='str'), instance_name=dict(type='str'), instance_id=dict(type='str'))

    def __init__(self):
        self._module = AnsibleModule(argument_spec=self._argument_spec, supports_check_mode=False)
        self._module_result = dict(failed=False)
        self._headers = {}
        self._headers['Content-Type'] = 'application/json'
        have_token = self._module.params['nitro_auth_token'] is not None
        have_userpass = None not in (self._module.params['nitro_user'], self._module.params['nitro_pass'])
        login_operation = self._module.params['operation'] == 'mas_login'
        if have_token and have_userpass:
            self.fail_module(msg='Cannot define both authentication token and username/password')
        if have_token:
            self._headers['Cookie'] = 'NITRO_AUTH_TOKEN=%s' % self._module.params['nitro_auth_token']
        if have_userpass and (not login_operation):
            self._headers['X-NITRO-USER'] = self._module.params['nitro_user']
            self._headers['X-NITRO-PASS'] = self._module.params['nitro_pass']
        if self._module.params['instance_ip'] is not None:
            self._headers['_MPS_API_PROXY_MANAGED_INSTANCE_IP'] = self._module.params['instance_ip']
        elif self._module.params['instance_name'] is not None:
            self._headers['_MPS_API_PROXY_MANAGED_INSTANCE_NAME'] = self._module.params['instance_name']
        elif self._module.params['instance_id'] is not None:
            self._headers['_MPS_API_PROXY_MANAGED_INSTANCE_ID'] = self._module.params['instance_id']

    def edit_response_data(self, r, info, result, success_status):
        if r is not None:
            result['http_response_body'] = codecs.decode(r.read(), 'utf-8')
        elif 'body' in info:
            result['http_response_body'] = codecs.decode(info['body'], 'utf-8')
            del info['body']
        else:
            result['http_response_body'] = ''
        result['http_response_data'] = info
        result['nitro_errorcode'] = None
        result['nitro_message'] = None
        result['nitro_severity'] = None
        if result['http_response_body'] != '':
            try:
                data = self._module.from_json(result['http_response_body'])
            except ValueError:
                data = {}
            result['nitro_errorcode'] = data.get('errorcode')
            result['nitro_message'] = data.get('message')
            result['nitro_severity'] = data.get('severity')
        if result['nitro_errorcode'] is None:
            if result['http_response_data'].get('status') != success_status:
                result['nitro_errorcode'] = -1
                result['nitro_message'] = result['http_response_data'].get('msg', 'HTTP status %s' % result['http_response_data']['status'])
                result['nitro_severity'] = 'ERROR'
            else:
                result['nitro_errorcode'] = 0
                result['nitro_message'] = 'Success'
                result['nitro_severity'] = 'NONE'

    def handle_get_return_object(self, result):
        result['nitro_object'] = []
        if result['nitro_errorcode'] == 0:
            if result['http_response_body'] != '':
                data = self._module.from_json(result['http_response_body'])
                if self._module.params['resource'] in data:
                    result['nitro_object'] = data[self._module.params['resource']]
        else:
            del result['nitro_object']

    def fail_module(self, msg, **kwargs):
        self._module_result['failed'] = True
        self._module_result['changed'] = False
        self._module_result.update(kwargs)
        self._module_result['msg'] = msg
        self._module.fail_json(**self._module_result)

    def main(self):
        if self._module.params['operation'] == 'add':
            result = self.add()
        if self._module.params['operation'] == 'update':
            result = self.update()
        if self._module.params['operation'] == 'delete':
            result = self.delete()
        if self._module.params['operation'] == 'delete_by_args':
            result = self.delete_by_args()
        if self._module.params['operation'] == 'get':
            result = self.get()
        if self._module.params['operation'] == 'get_by_args':
            result = self.get_by_args()
        if self._module.params['operation'] == 'get_filtered':
            result = self.get_filtered()
        if self._module.params['operation'] == 'get_all':
            result = self.get_all()
        if self._module.params['operation'] == 'count':
            result = self.count()
        if self._module.params['operation'] == 'mas_login':
            result = self.mas_login()
        if self._module.params['operation'] == 'action':
            result = self.action()
        if self._module.params['operation'] == 'save_config':
            result = self.save_config()
        if result['nitro_errorcode'] not in self._module.params['expected_nitro_errorcode']:
            self.fail_module(msg='NITRO Failure', **result)
        self._module_result.update(result)
        self._module.exit_json(**self._module_result)

    def exit_module(self):
        self._module.exit_json()

    def add(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['attributes'] is None:
            self.fail_module(msg='NITRO resource attributes are undefined.')
        url = '%s://%s/nitro/v1/config/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'])
        data = self._module.jsonify({self._module.params['resource']: self._module.params['attributes']})
        r, info = fetch_url(self._module, url=url, headers=self._headers, data=data, method='POST')
        result = {}
        self.edit_response_data(r, info, result, success_status=201)
        if result['nitro_errorcode'] == 0:
            self._module_result['changed'] = True
        else:
            self._module_result['changed'] = False
        return result

    def update(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['name'] is None:
            self.fail_module(msg='NITRO resource name is undefined.')
        if self._module.params['attributes'] is None:
            self.fail_module(msg='NITRO resource attributes are undefined.')
        url = '%s://%s/nitro/v1/config/%s/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'], self._module.params['name'])
        data = self._module.jsonify({self._module.params['resource']: self._module.params['attributes']})
        r, info = fetch_url(self._module, url=url, headers=self._headers, data=data, method='PUT')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        if result['nitro_errorcode'] == 0:
            self._module_result['changed'] = True
        else:
            self._module_result['changed'] = False
        return result

    def get(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['name'] is None:
            self.fail_module(msg='NITRO resource name is undefined.')
        url = '%s://%s/nitro/v1/config/%s/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'], self._module.params['name'])
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='GET')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        self.handle_get_return_object(result)
        self._module_result['changed'] = False
        return result

    def get_by_args(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['args'] is None:
            self.fail_module(msg='NITRO args is undefined.')
        url = '%s://%s/nitro/v1/config/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'])
        args_dict = self._module.params['args']
        args = ','.join(['%s:%s' % (k, args_dict[k]) for k in args_dict])
        args = 'args=' + args
        url = '?'.join([url, args])
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='GET')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        self.handle_get_return_object(result)
        self._module_result['changed'] = False
        return result

    def get_filtered(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['filter'] is None:
            self.fail_module(msg='NITRO filter is undefined.')
        filter_str = ','.join(('%s:%s' % (k, v) for k, v in self._module.params['filter'].items()))
        url = '%s://%s/nitro/v1/config/%s?filter=%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'], filter_str)
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='GET')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        self.handle_get_return_object(result)
        self._module_result['changed'] = False
        return result

    def get_all(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        url = '%s://%s/nitro/v1/config/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'])
        print('headers %s' % self._headers)
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='GET')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        self.handle_get_return_object(result)
        self._module_result['changed'] = False
        return result

    def delete(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['name'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        url = '%s://%s/nitro/v1/config/%s/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'], self._module.params['name'])
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='DELETE')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        if result['nitro_errorcode'] == 0:
            self._module_result['changed'] = True
        else:
            self._module_result['changed'] = False
        return result

    def delete_by_args(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['args'] is None:
            self.fail_module(msg='NITRO args is undefined.')
        url = '%s://%s/nitro/v1/config/%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'])
        args_dict = self._module.params['args']
        args = ','.join(['%s:%s' % (k, args_dict[k]) for k in args_dict])
        args = 'args=' + args
        url = '?'.join([url, args])
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='DELETE')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        if result['nitro_errorcode'] == 0:
            self._module_result['changed'] = True
        else:
            self._module_result['changed'] = False
        return result

    def count(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        url = '%s://%s/nitro/v1/config/%s?count=yes' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'])
        r, info = fetch_url(self._module, url=url, headers=self._headers, method='GET')
        result = {}
        self.edit_response_data(r, info, result)
        if result['http_response_body'] != '':
            data = self._module.from_json(result['http_response_body'])
            result['nitro_errorcode'] = data['errorcode']
            result['nitro_message'] = data['message']
            result['nitro_severity'] = data['severity']
            if self._module.params['resource'] in data:
                result['nitro_count'] = data[self._module.params['resource']][0]['__count']
        self._module_result['changed'] = False
        return result

    def action(self):
        if self._module.params['resource'] is None:
            self.fail_module(msg='NITRO resource is undefined.')
        if self._module.params['attributes'] is None:
            self.fail_module(msg='NITRO resource attributes are undefined.')
        if self._module.params['action'] is None:
            self.fail_module(msg='NITRO action is undefined.')
        url = '%s://%s/nitro/v1/config/%s?action=%s' % (self._module.params['nitro_protocol'], self._module.params['nsip'], self._module.params['resource'], self._module.params['action'])
        data = self._module.jsonify({self._module.params['resource']: self._module.params['attributes']})
        r, info = fetch_url(self._module, url=url, headers=self._headers, data=data, method='POST')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        if result['nitro_errorcode'] == 0:
            self._module_result['changed'] = True
        else:
            self._module_result['changed'] = False
        return result

    def mas_login(self):
        url = '%s://%s/nitro/v1/config/login' % (self._module.params['nitro_protocol'], self._module.params['nsip'])
        login_credentials = {'login': {'username': self._module.params['nitro_user'], 'password': self._module.params['nitro_pass']}}
        data = 'object=\n%s' % self._module.jsonify(login_credentials)
        r, info = fetch_url(self._module, url=url, headers=self._headers, data=data, method='POST')
        print(r, info)
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        if result['nitro_errorcode'] == 0:
            body_data = self._module.from_json(result['http_response_body'])
            result['nitro_auth_token'] = body_data['login'][0]['sessionid']
        self._module_result['changed'] = False
        return result

    def save_config(self):
        url = '%s://%s/nitro/v1/config/nsconfig?action=save' % (self._module.params['nitro_protocol'], self._module.params['nsip'])
        data = self._module.jsonify({'nsconfig': {}})
        r, info = fetch_url(self._module, url=url, headers=self._headers, data=data, method='POST')
        result = {}
        self.edit_response_data(r, info, result, success_status=200)
        self._module_result['changed'] = False
        return result