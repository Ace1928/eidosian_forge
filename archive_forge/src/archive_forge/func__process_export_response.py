from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _process_export_response(self, selector, response, schema_invt, log, export_path, url_params, params_schema):
    response_code = response[0]
    response_data = response[1]
    if response_code != 0 or 'data' not in response_data:
        log.write('\tno configuration data found\n')
        return
    if selector not in schema_invt:
        log.write('\trequested object has no corresponding ansible module\n')
        return
    state_present = schema_invt[selector]['stated']
    module_schema = schema_invt[selector]['options']
    remote_objects = response_data['data']
    counter = 0
    if type(remote_objects) is list:
        for remote_object in remote_objects:
            need_bypass = self.__fix_remote_object_internal(remote_object, module_schema, log)
            self._generate_playbook(counter, export_path, selector, remote_object, state_present, need_bypass, url_params, params_schema, log)
            counter += 1
    elif type(remote_objects) is dict:
        need_bypass = self.__fix_remote_object_internal(remote_objects, module_schema, log)
        self._generate_playbook(counter, export_path, selector, remote_objects, state_present, need_bypass, url_params, params_schema, log)
        counter += 1
    if not counter:
        self._nr_valid_selectors += 1