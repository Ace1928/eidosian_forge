from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def update_template_param(self, template, k, v):
    for i, param in enumerate(template['parameters']):
        if param['name'] == k:
            template['parameters'][i]['value'] = v
            return template
    return template