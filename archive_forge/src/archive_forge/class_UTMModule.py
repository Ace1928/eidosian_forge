from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class UTMModule(AnsibleModule):
    """
    This is a helper class to construct any UTM Module. This will automatically add the utm host, port, token,
    protocol, validate_certs and state field to the module. If you want to implement your own sophos utm module
    just initialize this UTMModule class and define the Payload fields that are needed for your module.
    See the other modules like utm_aaa_group for example.
    """

    def __init__(self, argument_spec, bypass_checks=False, no_log=False, mutually_exclusive=None, required_together=None, required_one_of=None, add_file_common_args=False, supports_check_mode=False, required_if=None):
        default_specs = dict(headers=dict(type='dict', required=False, default={}), utm_host=dict(type='str', required=True), utm_port=dict(type='int', default=4444), utm_token=dict(type='str', required=True, no_log=True), utm_protocol=dict(type='str', required=False, default='https', choices=['https', 'http']), validate_certs=dict(type='bool', required=False, default=True), state=dict(default='present', choices=['present', 'absent']))
        super(UTMModule, self).__init__(self._merge_specs(default_specs, argument_spec), bypass_checks, no_log, mutually_exclusive, required_together, required_one_of, add_file_common_args, supports_check_mode, required_if)

    def _merge_specs(self, default_specs, custom_specs):
        result = default_specs.copy()
        result.update(custom_specs)
        return result