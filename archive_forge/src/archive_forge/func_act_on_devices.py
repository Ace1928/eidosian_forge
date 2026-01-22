from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def act_on_devices(module, packet_conn, target_state):
    specified_identifiers = get_specified_device_identifiers(module)
    existing_devices = get_existing_devices(module, packet_conn)
    changed = False
    create_hostnames = []
    if target_state in ['present', 'active', 'rebooted']:
        existing_devices_names = [ed.hostname for ed in existing_devices]
        create_hostnames = [hn for hn in specified_identifiers['hostnames'] if hn not in existing_devices_names]
    process_devices = [d for d in existing_devices if d.id in specified_identifiers['ids'] or d.hostname in specified_identifiers['hostnames']]
    if target_state != 'present':
        _absent_state_map = {}
        for s in PACKET_DEVICE_STATES:
            _absent_state_map[s] = packet.Device.delete
        state_map = {'absent': _absent_state_map, 'active': {'inactive': packet.Device.power_on, 'provisioning': None, 'rebooting': None}, 'inactive': {'active': packet.Device.power_off}, 'rebooted': {'active': packet.Device.reboot, 'inactive': packet.Device.power_on, 'provisioning': None, 'rebooting': None}}
        for d in process_devices:
            if d.state == target_state:
                continue
            if d.state in state_map[target_state]:
                api_operation = state_map[target_state].get(d.state)
                if api_operation is not None:
                    api_operation(d)
                    changed = True
            else:
                _msg = "I don't know how to process existing device %s from state %s to state %s" % (d.hostname, d.state, target_state)
                raise Exception(_msg)
    created_devices = []
    if create_hostnames:
        created_devices = [create_single_device(module, packet_conn, n) for n in create_hostnames]
        if module.params.get('wait_for_public_IPv'):
            created_devices = wait_for_public_IPv(module, packet_conn, created_devices)
        changed = True
    processed_devices = created_devices + process_devices
    if target_state == 'active':
        processed_devices = wait_for_devices_active(module, packet_conn, processed_devices)
    return {'changed': changed, 'devices': [serialize_device(d) for d in processed_devices]}