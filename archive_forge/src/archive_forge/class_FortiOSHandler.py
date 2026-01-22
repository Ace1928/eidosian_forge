from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
class FortiOSHandler(object):

    def __init__(self, conn, mod, module_mkeyname=None):
        self._conn = conn
        self._module = mod
        self._mkeyname = module_mkeyname

    def _trace_to_string(self, trace):
        trace_string = ''
        for _trace in trace:
            trace_string += '%s%s' % (_trace, '.' if _trace != trace[-1] else '')
        return trace_string

    def _validate_member_parameter(self, trace, trace_param, trace_url_tokens, attr_blobs, attr_params):
        attr_blob = attr_blobs[0]
        current_attr_name = attr_blob['name']
        current_attr_mkey = attr_blob['mkey']
        trace.append(current_attr_name)
        if not attr_params:
            self._module.fail_json('parameter %s is empty' % self._trace_to_string(trace))
        if type(attr_params) not in [list, dict]:
            raise AssertionError('Invalid attribute type')
        if type(attr_params) is dict:
            trace_param_item = dict()
            trace_param_item[current_attr_name] = (None, attr_params)
            trace_param.append(trace_param_item)
            if len(attr_blobs) <= 1:
                raise AssertionError('Invalid attribute blob')
            next_attr_blob = attr_blobs[1]
            next_attr_name = next_attr_blob['name']
            self._validate_member_parameter(trace, trace_param, trace_url_tokens, attr_blobs[1:], attr_params[next_attr_name])
            del trace_param[-1]
            return
        for param in attr_params:
            if current_attr_mkey not in param or not param[current_attr_mkey]:
                self._module.fail_json('parameter %s.%s is empty' % (self._trace_to_string(trace), current_attr_mkey))
            trace_param_item = dict()
            trace_param_item[current_attr_name] = (param[current_attr_mkey], param)
            trace_param.append(trace_param_item)
            if len(attr_blobs) > 1:
                next_attr_blob = attr_blobs[1]
                next_attr_name = next_attr_blob['name']
                if next_attr_name in param:
                    self._validate_member_parameter(trace, trace_param, trace_url_tokens, attr_blobs[1:], param[next_attr_name])
                else:
                    url_tokens = list()
                    for token in trace_param:
                        url_tokens.append(token)
                    trace_url_tokens.append(url_tokens)
            else:
                url_tokens = list()
                for token in trace_param:
                    url_tokens.append(token)
                trace_url_tokens.append(url_tokens)
            del trace_param[-1]

    def _process_sub_object(self, all_urls, toplevel_url_token, traced_url_tokens, path, name):
        vdom = self._module.params['vdom'] if 'vdom' in self._module.params and self._module.params['vdom'] else None
        url_prefix = self.cmdb_url(path, name)
        url_suffix = ''
        if vdom == 'global':
            url_suffix = '?global=1'
        elif vdom:
            url_suffix = '?vdom=' + vdom
        for url_tokens in traced_url_tokens:
            url = dict()
            url_get = toplevel_url_token
            url_put = toplevel_url_token
            url_post = toplevel_url_token
            url_put_payload = dict()
            url_post_payload = dict()
            for token in url_tokens:
                token_name = str(list(token.keys())[0])
                token_value = str(token[token_name][0])
                token_payload = underscore_to_hyphen(token[token_name][1])
                token_islast = token == url_tokens[-1]
                if token[token_name][0]:
                    url_get += '/%s/%s' % (token_name.replace('_', '-'), urlencoding.quote(token_value, safe=''))
                    url_put += '/%s/%s' % (token_name.replace('_', '-'), urlencoding.quote(token_value, safe=''))
                else:
                    url_get += '/%s' % token_name.replace('_', '-')
                    url_put += '/%s' % token_name.replace('_', '-')
                if not token_islast:
                    if token[token_name][0]:
                        url_post += '/%s/%s' % (token_name.replace('_', '-'), urlencoding.quote(token_value, safe=''))
                    else:
                        url_post += '/%s' % token_name.replace('_', '-')
                else:
                    url_post += '/%s' % token_name.replace('_', '-')
                    url_post_payload = token_payload
                    url_put_payload = token_payload
            url['get'] = url_prefix + url_get + url_suffix
            url['put'] = url_prefix + url_put + url_suffix
            url['post'] = url_prefix + url_post + url_suffix
            url['put_payload'] = url_put_payload
            url['post_payload'] = url_post_payload
            url['delete'] = url['get']
            url['vdom'] = vdom
            all_urls.append(url)

    def _request_sub_object(self, sub_obj):
        directive_state = self._module.params['member_state']
        if directive_state not in ['present', 'absent']:
            raise AssertionError('Not invalid member_state directive.')
        status = None
        result_data = None
        if directive_state == 'present':
            status, result_data = self._conn.send_request(url=sub_obj['get'], params=None, method='GET')
            if status == 200:
                status, result_data = self._conn.send_request(url=sub_obj['put'], data=json.dumps(sub_obj['put_payload']), method='PUT')
                if status == 405:
                    status, result_data = self._conn.send_request(url=sub_obj['post'], data=json.dumps(sub_obj['post_payload']), method='POST')
            else:
                status, result_data = self._conn.send_request(url=sub_obj['post'], data=json.dumps(sub_obj['post_payload']), method='POST')
        else:
            status, result_data = self._conn.send_request(url=sub_obj['delete'], params=None, method='DELETE')
        result_data = self.formatresponse(result_data, status, vdom=sub_obj['vdom'])
        return result_data

    def _process_sub_object_result(self, results):
        meta = list()
        failed = False
        changed = False
        for result in results:
            sub_obj = result[0]
            result_data = result[1]
            url = sub_obj['get']
            suffix_index = url.find('?')
            if suffix_index >= 0:
                url = url[:suffix_index]
            result_data['object_path'] = url[12:]
            meta.append(result_data)
            if 'status' in result_data:
                if result_data['status'] == 'error':
                    failed = True
                elif result_data['status'] == 'success':
                    if 'revision_changed' in result_data and result_data['revision_changed'] is True:
                        changed = True
                    elif 'revision_changed' not in result_data:
                        changed = True
        self._module.exit_json(meta=meta, changed=changed, failed=failed)

    def do_member_operation(self, path, name):
        toplevel_name = (path + '_' + name).replace('-', '_').replace('.', '_').replace('+', 'plus')
        data = self._module.params
        if not data['member_state']:
            return
        if not data['member_path']:
            self._module.fail_json('member_path is empty while member_state is %s' % data['member_state'])
        attribute_path = list()
        for attr in data['member_path'].split('/'):
            if attr == '':
                continue
            attribute_path.append(attr.strip(' '))
        if not len(attribute_path):
            raise AssertionError('member_path should have at least one attribute')
        state_present = 'state' in data
        if state_present and (not self._mkeyname):
            raise AssertionError('Invalid mkey scheme!')
        if state_present and (not data[toplevel_name] or not data[toplevel_name][self._mkeyname]):
            raise AssertionError('parameter %s or %s.%s empty!' % (toplevel_name, toplevel_name, self._mkeyname))
        toplevel_url_token = ''
        if state_present:
            toplevel_url_token = '/%s' % data[toplevel_name][self._mkeyname]
        arg_spec = self._module.argument_spec[toplevel_name]['options']
        attr_spec = arg_spec
        attr_params = data[toplevel_name]
        if not attr_params:
            raise AssertionError('Parameter %s is empty' % toplevel_name)
        attr_blobs = list()
        for attr_pair in attribute_path:
            attr_pair_split = attr_pair.split(':')
            attr = attr_pair_split[0]
            if attr not in attr_spec:
                self._module.fail_json('Attribute %s not as part of module schema' % attr)
            attr_spec = attr_spec[attr]
            attr_type = attr_spec['type']
            if len(attr_pair_split) != 2 and attr_type != 'dict':
                self._module.fail_json('wrong attribute format: %s' % attr_pair)
            attr_mkey = attr_pair_split[1] if attr_type == 'list' else None
            if 'options' not in attr_spec:
                raise AssertionError('Attribute %s not member operable, no children options' % attr)
            attr_blob = dict()
            attr_blob['name'] = attr
            attr_blob['mkey'] = attr_mkey
            attr_blob['schema'] = attr_spec['options']
            attr_spec = attr_spec['options']
            attr_blobs.append(attr_blob)
        trace = list()
        trace_param = list()
        trace_url_tokens = list()
        urls = list()
        results = list()
        trace.append(toplevel_name)
        self._validate_member_parameter(trace, trace_param, trace_url_tokens, attr_blobs, attr_params[attr_blobs[0]['name']])
        self._process_sub_object(urls, toplevel_url_token, trace_url_tokens, path, name)
        for sub_obj in urls:
            result = self._request_sub_object(sub_obj)
            results.append((sub_obj, result))
        self._process_sub_object_result(results)

    def cmdb_url(self, path, name, vdom=None, mkey=None):
        url = '/api/v2/cmdb/' + path + '/' + name
        if mkey:
            url = url + '/' + urlencoding.quote(str(mkey), safe='')
        if vdom:
            if vdom == 'global':
                url += '?global=1'
            else:
                url += '?vdom=' + vdom
        return url

    def mon_url(self, path, name, vdom=None, mkey=None):
        url = '/api/v2/monitor/' + path + '/' + name
        if mkey:
            url = url + '/' + urlencoding.quote(str(mkey), safe='')
        if vdom:
            if vdom == 'global':
                url += '?global=1'
            else:
                url += '?vdom=' + vdom
        return url

    def log_url(self, path, name, mkey=None):
        url = '/api/v2/log/' + path + '/' + name
        if mkey:
            url = url + '/' + urlencoding.quote(str(mkey), safe='')
        return url

    def schema(self, path, name, vdom=None):
        if vdom is None:
            url = self.cmdb_url(path, name) + '?action=schema'
        else:
            url = self.cmdb_url(path, name, vdom=vdom) + '&action=schema'
        status, result_data = self._conn.send_request(url=url)
        if status == 200:
            if vdom == 'global':
                return json.loads(to_text(result_data))[0]['results']
            else:
                return json.loads(to_text(result_data))['results']
        else:
            return json.loads(to_text(result_data))

    def get_mkeyname(self, path, name, vdom=None):
        return self._mkeyname

    def get_mkey(self, path, name, data, vdom=None):
        keyname = self.get_mkeyname(path, name, vdom)
        if not keyname:
            return None
        else:
            try:
                mkey = data[keyname]
            except KeyError:
                return None
        return mkey

    def log_get(self, url, parameters=None):
        slash_index = url.find('/')
        full_url = self.log_url(url[:slash_index], url[slash_index + 1:])
        http_status, result_data = self._conn.send_request(url=full_url, params=parameters, method='GET')
        return self.formatresponse(result_data, http_status)

    def monitor_get(self, url, vdom=None, parameters=None):
        slash_index = url.find('/')
        full_url = self.mon_url(url[:slash_index], url[slash_index + 1:], vdom)
        http_status, result_data = self._conn.send_request(url=full_url, params=parameters, method='GET')
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def monitor_post(self, url, data=None, vdom=None, mkey=None, parameters=None):
        slash_index = url.find('/')
        url = self.mon_url(url[:slash_index], url[slash_index + 1:], vdom)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, data=json.dumps(data), method='POST')
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def get(self, path, name, vdom=None, mkey=None, parameters=None):
        url = self.cmdb_url(path, name, vdom, mkey=mkey)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, method='GET')
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def monitor(self, path, name, vdom=None, mkey=None, parameters=None):
        url = self.mon_url(path, name, vdom, mkey)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, method='GET')
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def set(self, path, name, data, mkey=None, vdom=None, parameters=None):
        if not mkey:
            mkey = self.get_mkey(path, name, data, vdom=vdom)
        url = self.cmdb_url(path, name, vdom, mkey)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, data=json.dumps(data), method='PUT')
        if parameters and 'action' in parameters and (parameters['action'] == 'move'):
            return self.formatresponse(result_data, http_status, vdom=vdom)
        if http_status == 404 or http_status == 405 or http_status == 500:
            return self.post(path, name, data, vdom, mkey)
        else:
            return self.formatresponse(result_data, http_status, vdom=vdom)

    def post(self, path, name, data, vdom=None, mkey=None, parameters=None):
        if mkey:
            mkeyname = self.get_mkeyname(path, name, vdom)
            data[mkeyname] = mkey
        url = self.cmdb_url(path, name, vdom, mkey=None)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, data=json.dumps(data), method='POST')
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def execute(self, path, name, data, vdom=None, mkey=None, parameters=None, timeout=300):
        url = self.mon_url(path, name, vdom, mkey=mkey)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, data=json.dumps(data), method='POST', timeout=timeout)
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def delete(self, path, name, vdom=None, mkey=None, parameters=None, data=None):
        if not mkey:
            mkey = self.get_mkey(path, name, data, vdom=vdom)
        url = self.cmdb_url(path, name, vdom, mkey)
        http_status, result_data = self._conn.send_request(url=url, params=parameters, data=json.dumps(data), method='DELETE')
        return self.formatresponse(result_data, http_status, vdom=vdom)

    def __to_local(self, data, http_status, is_array=False):
        try:
            resp = json.loads(data)
        except Exception:
            resp = {'raw': data}
        if is_array and type(resp) is not list:
            resp = [resp]
        if is_array and 'http_status' not in resp[0]:
            resp[0]['http_status'] = http_status
        elif not is_array and 'status' not in resp:
            resp['http_status'] = http_status
        return resp

    def formatresponse(self, res, http_status=500, vdom=None):
        if vdom == 'global':
            resp = self.__to_local(to_text(res), http_status, True)[0]
            resp['vdom'] = 'global'
        else:
            resp = self.__to_local(to_text(res), http_status, False)
        return resp

    def jsonraw(self, method, path, data, specific_params, vdom=None, parameters=None):
        url = path
        bvdom = False
        if vdom:
            if vdom == 'global':
                url += '?global=1'
            else:
                url += '?vdom=' + vdom
            bvdom = True
        if specific_params:
            if bvdom:
                url += '&'
            else:
                url += '?'
            url += specific_params
        if method == 'GET':
            http_status, result_data = self._conn.send_request(url=url, method='GET', params=parameters)
        else:
            http_status, result_data = self._conn.send_request(url=url, method=method, data=json.dumps(data), params=parameters)
        return self.formatresponse(result_data, http_status, vdom=vdom)