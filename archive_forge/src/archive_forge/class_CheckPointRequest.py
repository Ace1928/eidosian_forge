from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
class CheckPointRequest(object):

    def __init__(self, module=None, connection=None, headers=None, not_rest_data_keys=None, task_vars=None):
        self.module = module
        if module:
            self.connection = Connection(self.module._socket_path)
        elif connection:
            self.connection = connection
            try:
                self.connection.load_platform_plugins('check_point.mgmt.checkpoint')
                self.connection.set_options(var_options=task_vars)
            except ConnectionError:
                raise
        if not_rest_data_keys:
            self.not_rest_data_keys = not_rest_data_keys
        else:
            self.not_rest_data_keys = []
        self.not_rest_data_keys.append('validate_certs')
        self.headers = headers if headers else BASE_HEADERS

    def wait_for_task(self, version, connection, task_id):
        task_id_payload = {'task-id': task_id, 'details-level': 'full'}
        task_complete = False
        minutes_until_timeout = 30
        max_num_iterations = minutes_until_timeout * 30
        current_iteration = 0
        while not task_complete and current_iteration < max_num_iterations:
            current_iteration += 1
            code, response = send_request(connection, version, 'show-task', task_id_payload)
            attempts_counter = 0
            while code != 200:
                if attempts_counter < 5:
                    attempts_counter += 1
                    time.sleep(2)
                    code, response = send_request(connection, version, 'show-task', task_id_payload)
                else:
                    response['message'] = 'ERROR: Failed to handle asynchronous tasks as synchronous, tasks result is undefined. ' + response['message']
                    _fail_json(parse_fail_message(code, response))
            completed_tasks = 0
            for task in response['tasks']:
                if task['status'] == 'failed':
                    _fail_json('Task {0} with task id {1} failed. Look at the logs for more details'.format(task['task-name'], task['task-id']))
                if task['status'] == 'in progress':
                    break
                completed_tasks += 1
            if completed_tasks == len(response['tasks']):
                task_complete = True
            else:
                time.sleep(2)
        if not task_complete:
            _fail_json('ERROR: Timeout. Task-id: {0}.'.format(task_id_payload['task-id']))
        else:
            return response

    def discard_and_fail(self, code, response, connection, version, session_uid):
        discard_code, discard_response = send_request(connection, version, 'discard')
        if discard_code != 200:
            try:
                _fail_json(parse_fail_message(code, response) + ' Failed to discard session {0} with error {1} with message {2}'.format(session_uid, discard_code, discard_response))
            except Exception:
                _fail_json(parse_fail_message(code, response) + ' Failed to discard session with error {0} with message {1}'.format(discard_code, discard_response))
        _fail_json('Checkpoint session with ID: {0}'.format(session_uid) + ', ' + parse_fail_message(code, response) + ' Unpublished changes were discarded')

    def handle_publish(self, connection, version, payload):
        publish_code, publish_response = send_request(connection, version, 'publish')
        if publish_code != 200:
            self.discard_and_fail(publish_code, publish_response, connection, version)
        if payload.get('wait_for_task'):
            self.wait_for_task(version, connection, publish_response['task-id'])

    def handle_call(self, connection, version, api_url, payload, to_discard_on_failure, session_uid=None, to_publish=False):
        code, response = send_request(connection, version, api_url, payload)
        if code != 200:
            if to_discard_on_failure:
                self.discard_and_fail(code, response, connection, version, session_uid)
            elif 'object_not_found' not in response.get('code') and 'not found' not in response.get('message'):
                raise _fail_json('Checkpoint session with ID: {0}'.format(session_uid) + ', ' + parse_fail_message(code, response))
        elif 'wait_for_task' in payload and payload['wait_for_task']:
            if 'task-id' in response:
                response = self.wait_for_task(version, connection, response['task-id'])
            elif 'tasks' in response:
                for task in response['tasks']:
                    if 'task-id' in task:
                        task_id = task['task-id']
                        response[task_id] = self.wait_for_task(version, connection, task['task-id'])
                del response['tasks']
        if to_publish:
            self.handle_publish(connection, version, payload)
        return (code, response)

    def handle_add_and_set_result(self, connection, version, api_url, payload, session_uid, auto_publish_session=False):
        code, response = self.handle_call(connection, version, api_url, payload, True, session_uid, auto_publish_session)
        result = {'code': code, 'response': response, 'changed': True}
        return result

    def handle_delete(self, connection, payload, api_call_object, version):
        auto_publish = False
        payload_for_equals = {'type': api_call_object, 'params': payload}
        equals_code, equals_response = send_request(connection, version, 'equals', payload_for_equals)
        session_uid = connection.get_session_uid()
        if equals_code == 200:
            if payload.get('auto_publish_session'):
                auto_publish = payload['auto_publish_session']
                del payload['auto_publish_session']
            code, response = self.handle_call(connection, version, 'delete-' + api_call_object, payload, True, session_uid, auto_publish)
            result = {'code': code, 'response': response, 'changed': True}
        else:
            result = {'changed': False}
        if result.get('response'):
            result['checkpoint_session_uid'] = session_uid
        return result

    def api_call_facts(self, connection, payload, api_call_object, version):
        if payload.get('auto_publish_session'):
            del payload['auto_publish_session']
        code, response = self.handle_call(connection, version, api_call_object, payload, False)
        result = {'code': code, 'response': response}
        return result

    def api_call(self, connection, payload, remove_keys, api_call_object, state, equals_response, version, delete_params):
        result = {}
        auto_publish_session = False
        if payload.get('auto_publish_session'):
            auto_publish_session = payload['auto_publish_session']
            del payload['auto_publish_session']
        session_uid = connection.get_session_uid()
        if state == 'merged':
            if equals_response and equals_response.get('equals') is False:
                payload = remove_unwanted_key(payload, remove_keys)
                result = self.handle_add_and_set_result(connection, version, 'set-' + api_call_object, payload, session_uid, auto_publish_session)
            elif equals_response.get('code') or equals_response.get('message'):
                result = self.handle_add_and_set_result(connection, version, 'add-' + api_call_object, payload, session_uid, auto_publish_session)
        elif state == 'replaced':
            if equals_response and equals_response.get('equals') is False:
                code, response = self.handle_call(connection, version, 'delete-' + api_call_object, delete_params, True, session_uid, auto_publish_session)
                result = self.handle_add_and_set_result(connection, version, 'add-' + api_call_object, payload, session_uid, auto_publish_session)
            elif equals_response.get('code') or equals_response.get('message'):
                result = self.handle_add_and_set_result(connection, version, 'add-' + api_call_object, payload, session_uid, auto_publish_session)
        if result.get('response'):
            result['checkpoint_session_uid'] = session_uid
        return result

    def get_version(self, payload):
        return 'v' + payload['version'] + '/' if payload.get('version') else ''

    def _httpapi_error_handle(self, api_obj, state, **kwargs):
        try:
            result = {}
            version = self.get_version(kwargs['data'])
            if state == 'gathered':
                result = self.api_call_facts(self.connection, kwargs['data'], 'show-' + api_obj, version)
            elif state == 'deleted':
                result = self.handle_delete(self.connection, kwargs['data'], api_obj, version)
            elif state == 'merged' or state == 'replaced':
                payload_for_equals = {'type': api_obj, 'params': kwargs['data']}
                equals_code, equals_response = send_request(self.connection, version, 'equals', payload_for_equals)
                if equals_response.get('equals'):
                    result = {'code': equals_code, 'response': equals_response, 'changed': False}
                else:
                    result = self.api_call(self.connection, kwargs['data'], kwargs['remove_keys'], api_obj, state, equals_response, version, kwargs['delete_params'])
        except ConnectionError as e:
            raise _fail_json('connection error occurred: {0}'.format(e))
        except CertificateError as e:
            raise _fail_json('certificate error occurred: {0}'.format(e))
        except ValueError as e:
            raise _fail_json('certificate not found: {0}'.format(e))
        return result

    def post(self, obj, state, **kwargs):
        return self._httpapi_error_handle(obj, state, **kwargs)