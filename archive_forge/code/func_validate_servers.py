from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.network.a10.a10 import (axapi_call, a10_argument_spec, axapi_authenticate,
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec
def validate_servers(module, servers):
    for item in servers:
        for key in item:
            if key not in VALID_SERVER_FIELDS:
                module.fail_json(msg='invalid server field (%s), must be one of: %s' % (key, ','.join(VALID_SERVER_FIELDS)))
        if 'server' not in item:
            module.fail_json(msg='server definitions must define the server field')
        if 'port' in item:
            try:
                item['port'] = int(item['port'])
            except Exception:
                module.fail_json(msg='server port definitions must be integers')
        else:
            module.fail_json(msg='server definitions must define the port field')
        if 'status' in item:
            item['status'] = axapi_enabled_disabled(item['status'])
        else:
            item['status'] = 1