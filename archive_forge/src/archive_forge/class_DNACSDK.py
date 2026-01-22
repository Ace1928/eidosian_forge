from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
class DNACSDK(object):

    def __init__(self, params):
        self.result = dict(changed=False, result='')
        self.validate_response_schema = params.get('validate_response_schema')
        if DNAC_SDK_IS_INSTALLED:
            self.api = api.DNACenterAPI(username=params.get('dnac_username'), password=params.get('dnac_password'), base_url='https://{dnac_host}:{dnac_port}'.format(dnac_host=params.get('dnac_host'), dnac_port=params.get('dnac_port')), version=params.get('dnac_version'), verify=params.get('dnac_verify'), debug=params.get('dnac_debug'))
            if params.get('dnac_debug') and LOGGING_IN_STANDARD:
                logging.getLogger('dnacentersdk').addHandler(logging.StreamHandler())
        else:
            self.fail_json(msg="DNA Center Python SDK is not installed. Execute 'pip install dnacentersdk'")

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

    def _exec(self, family, function, params=None, op_modifies=False, **kwargs):
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
                if not self.validate_response_schema and op_modifies:
                    params['active_validation'] = False
                response = func(**params)
            else:
                response = func()
        except exceptions.dnacentersdkException as e:
            self.fail_json(msg='An error occured when executing operation. The error was: {error}'.format(error=to_native(e)))
        return response

    def fail_json(self, msg, **kwargs):
        self.result.update(**kwargs)
        raise Exception(msg)

    def exit_json(self):
        return self.result