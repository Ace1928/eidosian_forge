from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
class CloudflareAPI(object):
    cf_api_endpoint = 'https://api.cloudflare.com/client/v4'
    changed = False

    def __init__(self, module):
        self.module = module
        self.api_token = module.params['api_token']
        self.account_api_key = module.params['account_api_key']
        self.account_email = module.params['account_email']
        self.algorithm = module.params['algorithm']
        self.cert_usage = module.params['cert_usage']
        self.hash_type = module.params['hash_type']
        self.flag = module.params['flag']
        self.tag = module.params['tag']
        self.key_tag = module.params['key_tag']
        self.port = module.params['port']
        self.priority = module.params['priority']
        self.proto = lowercase_string(module.params['proto'])
        self.proxied = module.params['proxied']
        self.selector = module.params['selector']
        self.record = lowercase_string(module.params['record'])
        self.service = lowercase_string(module.params['service'])
        self.is_solo = module.params['solo']
        self.state = module.params['state']
        self.timeout = module.params['timeout']
        self.ttl = module.params['ttl']
        self.type = module.params['type']
        self.value = module.params['value']
        self.weight = module.params['weight']
        self.zone = lowercase_string(module.params['zone'])
        if self.record == '@':
            self.record = self.zone
        if self.type in ['CNAME', 'NS', 'MX', 'SRV'] and self.value is not None:
            self.value = self.value.rstrip('.').lower()
        if self.type == 'AAAA' and self.value is not None:
            self.value = self.value.lower()
        if self.type == 'SRV':
            if self.proto is not None and (not self.proto.startswith('_')):
                self.proto = '_' + self.proto
            if self.service is not None and (not self.service.startswith('_')):
                self.service = '_' + self.service
        if self.type == 'TLSA':
            if self.proto is not None and (not self.proto.startswith('_')):
                self.proto = '_' + self.proto
            if self.port is not None:
                self.port = '_' + str(self.port)
        if not self.record.endswith(self.zone):
            self.record = self.record + '.' + self.zone
        if self.type == 'DS':
            if self.record == self.zone:
                self.module.fail_json(msg='DS records only apply to subdomains.')

    def _cf_simple_api_call(self, api_call, method='GET', payload=None):
        if self.api_token:
            headers = {'Authorization': 'Bearer ' + self.api_token, 'Content-Type': 'application/json'}
        else:
            headers = {'X-Auth-Email': self.account_email, 'X-Auth-Key': self.account_api_key, 'Content-Type': 'application/json'}
        data = None
        if payload:
            try:
                data = json.dumps(payload)
            except Exception as e:
                self.module.fail_json(msg='Failed to encode payload as JSON: %s ' % to_native(e))
        resp, info = fetch_url(self.module, self.cf_api_endpoint + api_call, headers=headers, data=data, method=method, timeout=self.timeout)
        if info['status'] not in [200, 304, 400, 401, 403, 429, 405, 415]:
            self.module.fail_json(msg='Failed API call {0}; got unexpected HTTP code {1}: {2}'.format(api_call, info['status'], info.get('msg')))
        error_msg = ''
        if info['status'] == 401:
            error_msg = 'API user does not have permission; Status: {0}; Method: {1}: Call: {2}'.format(info['status'], method, api_call)
        elif info['status'] == 403:
            error_msg = 'API request not authenticated; Status: {0}; Method: {1}: Call: {2}'.format(info['status'], method, api_call)
        elif info['status'] == 429:
            error_msg = 'API client is rate limited; Status: {0}; Method: {1}: Call: {2}'.format(info['status'], method, api_call)
        elif info['status'] == 405:
            error_msg = 'API incorrect HTTP method provided; Status: {0}; Method: {1}: Call: {2}'.format(info['status'], method, api_call)
        elif info['status'] == 415:
            error_msg = 'API request is not valid JSON; Status: {0}; Method: {1}: Call: {2}'.format(info['status'], method, api_call)
        elif info['status'] == 400:
            error_msg = 'API bad request; Status: {0}; Method: {1}: Call: {2}'.format(info['status'], method, api_call)
        result = None
        try:
            content = resp.read()
        except AttributeError:
            if info['body']:
                content = info['body']
            else:
                error_msg += '; The API response was empty'
        if content:
            try:
                result = json.loads(to_text(content, errors='surrogate_or_strict'))
            except getattr(json, 'JSONDecodeError', ValueError) as e:
                error_msg += '; Failed to parse API response with error {0}: {1}'.format(to_native(e), content)
        if result is None:
            self.module.fail_json(msg=error_msg)
        if 'success' not in result:
            error_msg += '; Unexpected error details: {0}'.format(result.get('error'))
            self.module.fail_json(msg=error_msg)
        if not result['success']:
            error_msg += '; Error details: '
            for error in result['errors']:
                error_msg += 'code: {0}, error: {1}; '.format(error['code'], error['message'])
                if 'error_chain' in error:
                    for chain_error in error['error_chain']:
                        error_msg += 'code: {0}, error: {1}; '.format(chain_error['code'], chain_error['message'])
            self.module.fail_json(msg=error_msg)
        return (result, info['status'])

    def _cf_api_call(self, api_call, method='GET', payload=None):
        result, status = self._cf_simple_api_call(api_call, method, payload)
        data = result['result']
        if 'result_info' in result:
            pagination = result['result_info']
            if pagination['total_pages'] > 1:
                next_page = int(pagination['page']) + 1
                parameters = ['page={0}'.format(next_page)]
                if '?' in api_call:
                    raw_api_call, query = api_call.split('?', 1)
                    parameters += [param for param in query.split('&') if not param.startswith('page')]
                else:
                    raw_api_call = api_call
                while next_page <= pagination['total_pages']:
                    raw_api_call += '?' + '&'.join(parameters)
                    result, status = self._cf_simple_api_call(raw_api_call, method, payload)
                    data += result['result']
                    next_page += 1
        return (data, status)

    def _get_zone_id(self, zone=None):
        if not zone:
            zone = self.zone
        zones = self.get_zones(zone)
        if len(zones) > 1:
            self.module.fail_json(msg='More than one zone matches {0}'.format(zone))
        if len(zones) < 1:
            self.module.fail_json(msg='No zone found with name {0}'.format(zone))
        return zones[0]['id']

    def get_zones(self, name=None):
        if not name:
            name = self.zone
        param = ''
        if name:
            param = '?' + urlencode({'name': name})
        zones, status = self._cf_api_call('/zones' + param)
        return zones

    def get_dns_records(self, zone_name=None, type=None, record=None, value=''):
        if not zone_name:
            zone_name = self.zone
        if not type:
            type = self.type
        if not record:
            record = self.record
        if not value and value is not None:
            value = self.value
        zone_id = self._get_zone_id()
        api_call = '/zones/{0}/dns_records'.format(zone_id)
        query = {}
        if type:
            query['type'] = type
        if record:
            query['name'] = record
        if value:
            query['content'] = value
        if query:
            api_call += '?' + urlencode(query)
        records, status = self._cf_api_call(api_call)
        return records

    def delete_dns_records(self, **kwargs):
        params = {}
        for param in ['port', 'proto', 'service', 'solo', 'type', 'record', 'value', 'weight', 'zone', 'algorithm', 'cert_usage', 'hash_type', 'selector', 'key_tag', 'flag', 'tag']:
            if param in kwargs:
                params[param] = kwargs[param]
            else:
                params[param] = getattr(self, param)
        records = []
        content = params['value']
        search_record = params['record']
        if params['type'] == 'SRV':
            if not (params['value'] is None or params['value'] == ''):
                content = str(params['weight']) + '\t' + str(params['port']) + '\t' + params['value']
            search_record = params['service'] + '.' + params['proto'] + '.' + params['record']
        elif params['type'] == 'DS':
            if not (params['value'] is None or params['value'] == ''):
                content = str(params['key_tag']) + '\t' + str(params['algorithm']) + '\t' + str(params['hash_type']) + '\t' + params['value']
        elif params['type'] == 'SSHFP':
            if not (params['value'] is None or params['value'] == ''):
                content = str(params['algorithm']) + ' ' + str(params['hash_type']) + ' ' + params['value'].upper()
        elif params['type'] == 'TLSA':
            if not (params['value'] is None or params['value'] == ''):
                content = str(params['cert_usage']) + '\t' + str(params['selector']) + '\t' + str(params['hash_type']) + '\t' + params['value']
            search_record = params['port'] + '.' + params['proto'] + '.' + params['record']
        if params['solo']:
            search_value = None
        else:
            search_value = content
        records = self.get_dns_records(params['zone'], params['type'], search_record, search_value)
        for rr in records:
            if params['solo']:
                if not (rr['type'] == params['type'] and rr['name'] == search_record and (rr['content'] == content)):
                    self.changed = True
                    if not self.module.check_mode:
                        result, info = self._cf_api_call('/zones/{0}/dns_records/{1}'.format(rr['zone_id'], rr['id']), 'DELETE')
            else:
                self.changed = True
                if not self.module.check_mode:
                    result, info = self._cf_api_call('/zones/{0}/dns_records/{1}'.format(rr['zone_id'], rr['id']), 'DELETE')
        return self.changed

    def ensure_dns_record(self, **kwargs):
        params = {}
        for param in ['port', 'priority', 'proto', 'proxied', 'service', 'ttl', 'type', 'record', 'value', 'weight', 'zone', 'algorithm', 'cert_usage', 'hash_type', 'selector', 'key_tag', 'flag', 'tag']:
            if param in kwargs:
                params[param] = kwargs[param]
            else:
                params[param] = getattr(self, param)
        search_value = params['value']
        search_record = params['record']
        new_record = None
        if params['type'] is None or params['record'] is None:
            self.module.fail_json(msg='You must provide a type and a record to create a new record')
        if params['type'] in ['A', 'AAAA', 'CNAME', 'TXT', 'MX', 'NS', 'SPF']:
            if not params['value']:
                self.module.fail_json(msg='You must provide a non-empty value to create this record type')
            if params['type'] == 'CNAME':
                search_value = None
            new_record = {'type': params['type'], 'name': params['record'], 'content': params['value'], 'ttl': params['ttl']}
        if params['type'] in ['A', 'AAAA', 'CNAME']:
            new_record['proxied'] = params['proxied']
        if params['type'] == 'MX':
            for attr in [params['priority'], params['value']]:
                if attr is None or attr == '':
                    self.module.fail_json(msg='You must provide priority and a value to create this record type')
            new_record = {'type': params['type'], 'name': params['record'], 'content': params['value'], 'priority': params['priority'], 'ttl': params['ttl']}
        if params['type'] == 'SRV':
            for attr in [params['port'], params['priority'], params['proto'], params['service'], params['weight'], params['value']]:
                if attr is None or attr == '':
                    self.module.fail_json(msg='You must provide port, priority, proto, service, weight and a value to create this record type')
            srv_data = {'target': params['value'], 'port': params['port'], 'weight': params['weight'], 'priority': params['priority'], 'name': params['record'], 'proto': params['proto'], 'service': params['service']}
            new_record = {'type': params['type'], 'ttl': params['ttl'], 'data': srv_data}
            search_value = str(params['weight']) + '\t' + str(params['port']) + '\t' + params['value']
            search_record = params['service'] + '.' + params['proto'] + '.' + params['record']
        if params['type'] == 'DS':
            for attr in [params['key_tag'], params['algorithm'], params['hash_type'], params['value']]:
                if attr is None or attr == '':
                    self.module.fail_json(msg='You must provide key_tag, algorithm, hash_type and a value to create this record type')
            ds_data = {'key_tag': params['key_tag'], 'algorithm': params['algorithm'], 'digest_type': params['hash_type'], 'digest': params['value']}
            new_record = {'type': params['type'], 'name': params['record'], 'data': ds_data, 'ttl': params['ttl']}
            search_value = str(params['key_tag']) + '\t' + str(params['algorithm']) + '\t' + str(params['hash_type']) + '\t' + params['value']
        if params['type'] == 'SSHFP':
            for attr in [params['algorithm'], params['hash_type'], params['value']]:
                if attr is None or attr == '':
                    self.module.fail_json(msg='You must provide algorithm, hash_type and a value to create this record type')
            sshfp_data = {'fingerprint': params['value'].upper(), 'type': params['hash_type'], 'algorithm': params['algorithm']}
            new_record = {'type': params['type'], 'name': params['record'], 'data': sshfp_data, 'ttl': params['ttl']}
            search_value = str(params['algorithm']) + ' ' + str(params['hash_type']) + ' ' + params['value']
        if params['type'] == 'TLSA':
            for attr in [params['port'], params['proto'], params['cert_usage'], params['selector'], params['hash_type'], params['value']]:
                if attr is None or attr == '':
                    self.module.fail_json(msg='You must provide port, proto, cert_usage, selector, hash_type and a value to create this record type')
            search_record = params['port'] + '.' + params['proto'] + '.' + params['record']
            tlsa_data = {'usage': params['cert_usage'], 'selector': params['selector'], 'matching_type': params['hash_type'], 'certificate': params['value']}
            new_record = {'type': params['type'], 'name': search_record, 'data': tlsa_data, 'ttl': params['ttl']}
            search_value = str(params['cert_usage']) + '\t' + str(params['selector']) + '\t' + str(params['hash_type']) + '\t' + params['value']
        if params['type'] == 'CAA':
            for attr in [params['flag'], params['tag'], params['value']]:
                if attr is None or attr == '':
                    self.module.fail_json(msg='You must provide flag, tag and a value to create this record type')
            caa_data = {'flags': params['flag'], 'tag': params['tag'], 'value': params['value']}
            new_record = {'type': params['type'], 'name': params['record'], 'data': caa_data, 'ttl': params['ttl']}
            search_value = None
        zone_id = self._get_zone_id(params['zone'])
        records = self.get_dns_records(params['zone'], params['type'], search_record, search_value)
        if len(records) > 1:
            if params['type'] == 'CAA':
                for rr in records:
                    if rr['data']['flags'] == caa_data['flags'] and rr['data']['tag'] == caa_data['tag'] and (rr['data']['value'] == caa_data['value']):
                        return (rr, self.changed)
            else:
                self.module.fail_json(msg='More than one record already exists for the given attributes. That should be impossible, please open an issue!')
        if len(records) == 1:
            cur_record = records[0]
            do_update = False
            if params['ttl'] is not None and cur_record['ttl'] != params['ttl']:
                do_update = True
            if params['priority'] is not None and 'priority' in cur_record and (cur_record['priority'] != params['priority']):
                do_update = True
            if 'proxied' in new_record and 'proxied' in cur_record and (cur_record['proxied'] != params['proxied']):
                do_update = True
            if 'data' in new_record and 'data' in cur_record:
                if cur_record['data'] != new_record['data']:
                    do_update = True
            if params['type'] == 'CNAME' and cur_record['content'] != new_record['content']:
                do_update = True
            if do_update:
                if self.module.check_mode:
                    result = new_record
                else:
                    result, info = self._cf_api_call('/zones/{0}/dns_records/{1}'.format(zone_id, records[0]['id']), 'PUT', new_record)
                self.changed = True
                return (result, self.changed)
            else:
                return (records, self.changed)
        if self.module.check_mode:
            result = new_record
        else:
            result, info = self._cf_api_call('/zones/{0}/dns_records'.format(zone_id), 'POST', new_record)
        self.changed = True
        return (result, self.changed)