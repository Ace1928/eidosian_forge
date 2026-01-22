from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _transform_ip_list(self, resource):
    """ Workaround for 4.11 return API break """
    keys = ['endip', 'startip']
    if resource:
        for key in keys:
            if key in resource and isinstance(resource[key], list):
                resource[key] = resource[key][0]
    return resource