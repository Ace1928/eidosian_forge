from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def updated_record(module):
    return {'kind': 'dns#resourceRecordSet', 'name': module.params['name'], 'type': module.params['type'], 'ttl': module.params['ttl'] if module.params['ttl'] else 900, 'rrdatas': module.params['target']}