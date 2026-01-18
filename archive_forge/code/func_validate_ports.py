from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.network.a10.a10 import (axapi_call, a10_argument_spec, axapi_authenticate, axapi_failure,
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec
def validate_ports(module, ports):
    for item in ports:
        for key in item:
            if key not in VALID_PORT_FIELDS:
                module.fail_json(msg='invalid port field (%s), must be one of: %s' % (key, ','.join(VALID_PORT_FIELDS)))
        if 'port_num' in item:
            try:
                item['port_num'] = int(item['port_num'])
            except Exception:
                module.fail_json(msg='port_num entries in the port definitions must be integers')
        else:
            module.fail_json(msg='port definitions must define the port_num field')
        if 'protocol' in item:
            protocol = axapi_get_port_protocol(item['protocol'])
            if not protocol:
                module.fail_json(msg='invalid port protocol, must be one of: %s' % ','.join(AXAPI_PORT_PROTOCOLS))
            else:
                item['protocol'] = protocol
        else:
            module.fail_json(msg='port definitions must define the port protocol (%s)' % ','.join(AXAPI_PORT_PROTOCOLS))
        if 'status' in item:
            item['status'] = axapi_enabled_disabled(item['status'])
        else:
            item['status'] = 1