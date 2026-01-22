from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
class MERAKI(object):

    def __init__(self, params):
        self.result = dict(changed=False, result='')
        if MERAKI_SDK_IS_INSTALLED:
            self.api = meraki.DashboardAPI(api_key=params.get('meraki_api_key'), base_url=params.get('meraki_base_url'), single_request_timeout=params.get('meraki_single_request_timeout'), certificate_path=params.get('meraki_certificate_path'), requests_proxy=params.get('meraki_requests_proxy'), wait_on_rate_limit=params.get('meraki_wait_on_rate_limit'), nginx_429_retry_wait_time=params.get('meraki_nginx_429_retry_wait_time'), action_batch_retry_wait_time=params.get('meraki_action_batch_retry_wait_time'), retry_4xx_error=params.get('meraki_retry_4xx_error'), retry_4xx_error_wait_time=params.get('meraki_retry_4xx_error_wait_time'), maximum_retries=params.get('meraki_maximum_retries'), output_log=params.get('meraki_output_log'), log_path=params.get('meraki_log_path'), log_file_prefix=params.get('meraki_log_file_prefix'), print_console=params.get('meraki_print_console'), suppress_logging=params.get('meraki_suppress_logging'), simulate=params.get('meraki_simulate'), be_geo_id=params.get('meraki_be_geo_id'), caller='MerakiAnsibleCollection/1.0.0 Cisco', use_iterator_for_get_pages=params.get('meraki_use_iterator_for_get_pages'), inherit_logging_config=params.get('meraki_inherit_logging_config'))
        else:
            self.fail_json(msg="Meraki SDK is not installed. Execute 'pip install meraki'")

    def changed(self):
        self.result['changed'] = True

    def object_created(self):
        self.changed()
        self.result['result'] = 'Object created'

    def object_updated(self):
        self.changed()
        self.result['result'] = 'Object updated'

    def object_deleted(self):
        self.changed()
        self.result['result'] = 'Object deleted'

    def object_already_absent(self):
        self.result['result'] = 'Object already absent'

    def object_already_present(self):
        self.result['result'] = 'Object already present'

    def object_present_and_different(self):
        self.result['result'] = 'Object already present, but it has different values to the requested'

    def object_modify_result(self, changed=None, result=None):
        if result is not None:
            self.result['result'] = result
        if changed:
            self.changed()

    def is_file(self, file_path):
        return os.path.isfile(file_path)

    def extract_file_name(self, file_path):
        return os.path.basename(file_path)

    def exec_meraki(self, family, function, params=None, op_modifies=False, **kwargs):
        try:
            family = getattr(self.api, family)
            func = getattr(family, function)
        except Exception as e:
            self.fail_json(msg=e)
        try:
            if params:
                file_paths_params = kwargs.get('file_paths', [])
                if file_paths_params and isinstance(file_paths_params, list):
                    multipart_fields = {}
                    for key, value in file_paths_params:
                        if isinstance(params.get(key), str) and self.is_file(params[key]):
                            file_name = self.extract_file_name(params[key])
                            file_path = params[key]
                            multipart_fields[value] = (file_name, open(file_path, 'rb'))
                    params.setdefault('multipart_fields', multipart_fields)
                    params.setdefault('multipart_monitor_callback', None)
                response = func(**params)
            else:
                response = func()
        except exceptions.APIError as e:
            self.fail_json(msg='An error occured when executing operation.The error was: {error}'.format(error=to_native(e)))
        return response

    def fail_json(self, msg, **kwargs):
        self.result.update(**kwargs)
        raise AnsibleActionFail(msg, kwargs)

    def exit_json(self):
        return self.result

    def verify_array(self, verify_interface, **kwargs):
        if verify_interface is None:
            return list()
        if isinstance(verify_interface, list):
            if len(verify_interface) == 0:
                return list()
            if verify_interface[0] is None:
                return list()
        return verify_interface