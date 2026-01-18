from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def user_input_parameters(module):
    return {'ip_address': module.params.get('ip_address'), 'subnet_id': module.params.get('subnet_id')}