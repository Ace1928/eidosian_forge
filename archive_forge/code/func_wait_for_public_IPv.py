from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def wait_for_public_IPv(module, packet_conn, created_devices):

    def has_public_ip(addr_list, ip_v):
        return any((a['public'] and a['address_family'] == ip_v and a['address'] for a in addr_list))

    def all_have_public_ip(ds, ip_v):
        return all((has_public_ip(d.ip_addresses, ip_v) for d in ds))
    address_family = module.params.get('wait_for_public_IPv')
    wait_timeout = module.params.get('wait_timeout')
    wait_timeout = time.time() + wait_timeout
    while wait_timeout > time.time():
        refreshed = refresh_device_list(module, packet_conn, created_devices)
        if all_have_public_ip(refreshed, address_family):
            return refreshed
        time.sleep(5)
    raise Exception('Waiting for IPv%d address timed out. Hostnames: %s' % (address_family, [d.hostname for d in created_devices]))