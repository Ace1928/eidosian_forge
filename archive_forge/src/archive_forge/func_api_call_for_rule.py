from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def api_call_for_rule(module, api_call_object):
    is_access_rule = True if 'access' in api_call_object else False
    payload = get_payload_from_parameters(module.params)
    connection = Connection(module._socket_path)
    version = get_version(module)
    result = {'changed': False}
    if module.check_mode:
        return result
    if is_access_rule:
        copy_payload_without_some_params = extract_payload_without_some_params(payload, ['action', 'position', 'search_entire_rulebase'])
    else:
        copy_payload_without_some_params = extract_payload_without_some_params(payload, ['position'])
    payload_for_equals = {'type': api_call_object, 'params': copy_payload_without_some_params}
    equals_code, equals_response = send_request(connection, version, 'equals', payload_for_equals)
    result['checkpoint_session_uid'] = connection.get_session_uid()
    handle_equals_failure(module, equals_code, equals_response)
    if module.params['state'] == 'present':
        if equals_code == 200:
            if equals_response['equals']:
                if not is_equals_with_all_params(payload, connection, version, api_call_object, is_access_rule):
                    equals_response['equals'] = False
            if not equals_response['equals']:
                if 'position' in payload:
                    payload['new-position'] = payload['position']
                    del payload['position']
                if 'search-entire-rulebase' in payload:
                    del payload['search-entire-rulebase']
                handle_call_and_set_result(connection, version, 'set-' + api_call_object, payload, module, result)
        elif equals_code == 404:
            if 'search-entire-rulebase' in payload:
                del payload['search-entire-rulebase']
            handle_call_and_set_result(connection, version, 'add-' + api_call_object, payload, module, result)
    elif module.params['state'] == 'absent':
        handle_delete(equals_code, payload, delete_params, connection, version, api_call_object, module, result)
    return result