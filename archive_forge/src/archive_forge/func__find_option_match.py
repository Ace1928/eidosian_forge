from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _find_option_match(self, template, param_name, internal_name=None):
    if not internal_name:
        internal_name = param_name
    if param_name in self.module.params.get('template_find_options'):
        param_value = self.module.params.get(param_name)
        if not param_value:
            self.fail_json(msg='The param template_find_options has %s but param was not provided.' % param_name)
        if template[internal_name] == param_value:
            return True
    return False