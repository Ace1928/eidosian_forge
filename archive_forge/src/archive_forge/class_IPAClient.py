from __future__ import (absolute_import, division, print_function)
import json
import os
import socket
import uuid
import re
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, HAS_GSSAPI
from ansible.module_utils.basic import env_fallback, AnsibleFallbackNotFound
class IPAClient(object):

    def __init__(self, module, host, port, protocol):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.module = module
        self.headers = None
        self.timeout = module.params.get('ipa_timeout')
        self.use_gssapi = False

    def get_base_url(self):
        return '%s://%s/ipa' % (self.protocol, self.host)

    def get_json_url(self):
        return '%s/session/json' % self.get_base_url()

    def login(self, username, password):
        if 'KRB5CCNAME' in os.environ and HAS_GSSAPI:
            self.use_gssapi = True
        elif 'KRB5_CLIENT_KTNAME' in os.environ and HAS_GSSAPI:
            ccache = 'MEMORY:' + str(uuid.uuid4())
            os.environ['KRB5CCNAME'] = ccache
            self.use_gssapi = True
        else:
            if not password:
                if 'KRB5CCNAME' in os.environ or 'KRB5_CLIENT_KTNAME' in os.environ:
                    self.module.warn("In order to use GSSAPI, you need to install 'urllib_gssapi'")
                self._fail('login', 'Password is required if not using GSSAPI. To use GSSAPI, please set the KRB5_CLIENT_KTNAME or KRB5CCNAME (or both)  environment variables.')
            url = '%s/session/login_password' % self.get_base_url()
            data = 'user=%s&password=%s' % (quote(username, safe=''), quote(password, safe=''))
            headers = {'referer': self.get_base_url(), 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'text/plain'}
            try:
                resp, info = fetch_url(module=self.module, url=url, data=to_bytes(data), headers=headers, timeout=self.timeout)
                status_code = info['status']
                if status_code not in [200, 201, 204]:
                    self._fail('login', info['msg'])
                self.headers = {'Cookie': info.get('set-cookie')}
            except Exception as e:
                self._fail('login', to_native(e))
        if not self.headers:
            self.headers = dict()
        self.headers.update({'referer': self.get_base_url(), 'Content-Type': 'application/json', 'Accept': 'application/json'})

    def _fail(self, msg, e):
        if 'message' in e:
            err_string = e.get('message')
        else:
            err_string = e
        self.module.fail_json(msg='%s: %s' % (msg, err_string))

    def get_ipa_version(self):
        response = self.ping()['summary']
        ipa_ver_regex = re.compile('IPA server version (\\d\\.\\d\\.\\d).*')
        version_match = ipa_ver_regex.match(response)
        ipa_version = None
        if version_match:
            ipa_version = version_match.groups()[0]
        return ipa_version

    def ping(self):
        return self._post_json(method='ping', name=None)

    def _post_json(self, method, name, item=None):
        if item is None:
            item = {}
        url = '%s/session/json' % self.get_base_url()
        data = dict(method=method)
        if method in ('ping', 'config_show', 'otpconfig_show'):
            data['params'] = [[], {}]
        elif method in ('config_mod', 'otpconfig_mod'):
            data['params'] = [[], item]
        else:
            data['params'] = [[name], item]
        try:
            resp, info = fetch_url(module=self.module, url=url, data=to_bytes(json.dumps(data)), headers=self.headers, timeout=self.timeout, use_gssapi=self.use_gssapi)
            status_code = info['status']
            if status_code not in [200, 201, 204]:
                self._fail(method, info['msg'])
        except Exception as e:
            self._fail('post %s' % method, to_native(e))
        if PY3:
            charset = resp.headers.get_content_charset('latin-1')
        else:
            response_charset = resp.headers.getparam('charset')
            if response_charset:
                charset = response_charset
            else:
                charset = 'latin-1'
        resp = json.loads(to_text(resp.read(), encoding=charset))
        err = resp.get('error')
        if err is not None:
            self._fail('response %s' % method, err)
        if 'result' in resp:
            result = resp.get('result')
            if 'result' in result:
                result = result.get('result')
                if isinstance(result, list):
                    if len(result) > 0:
                        return result[0]
                    else:
                        return {}
            return result
        return None

    def get_diff(self, ipa_data, module_data):
        result = []
        for key in module_data.keys():
            mod_value = module_data.get(key, None)
            if isinstance(mod_value, list):
                default = []
            else:
                default = None
            ipa_value = ipa_data.get(key, default)
            if isinstance(ipa_value, list) and (not isinstance(mod_value, list)):
                mod_value = [mod_value]
            if isinstance(ipa_value, list) and isinstance(mod_value, list):
                mod_value = sorted(mod_value)
                ipa_value = sorted(ipa_value)
            if mod_value != ipa_value:
                result.append(key)
        return result

    def modify_if_diff(self, name, ipa_list, module_list, add_method, remove_method, item=None, append=None):
        changed = False
        diff = list(set(ipa_list) - set(module_list))
        if append is not True and len(diff) > 0:
            changed = True
            if not self.module.check_mode:
                if item:
                    remove_method(name=name, item={item: diff})
                else:
                    remove_method(name=name, item=diff)
        diff = list(set(module_list) - set(ipa_list))
        if len(diff) > 0:
            changed = True
            if not self.module.check_mode:
                if item:
                    add_method(name=name, item={item: diff})
                else:
                    add_method(name=name, item=diff)
        return changed