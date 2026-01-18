from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def unwrap_resource_filter(module):
    return {'name': module.params['name'], 'host': module.params['host']}