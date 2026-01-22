from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def act_on_volume_attachment(target_state, module, packet_conn):
    return_dict = {'changed': False}
    volspec = module.params.get('volume')
    devspec = module.params.get('device')
    if devspec is None and target_state == 'present':
        raise Exception('If you want to attach a volume, you must specify a device.')
    project_id = module.params.get('project_id')
    volumes_api_method = 'projects/{0}/storage'.format(project_id)
    volumes = packet_conn.call_api(volumes_api_method, params={'include': 'facility,attachments.device'})['volumes']
    v_match = get_volume_selector(volspec)
    matching_volumes = [v for v in volumes if v_match(v)]
    validate_selected(matching_volumes, 'volume', volspec)
    volume = matching_volumes[0]
    return_dict['volume_id'] = volume['id']
    device = None
    if devspec is not None:
        devices_api_method = 'projects/{0}/devices'.format(project_id)
        devices = packet_conn.call_api(devices_api_method)['devices']
        d_match = get_device_selector(devspec)
        matching_devices = [d for d in devices if d_match(d)]
        validate_selected(matching_devices, 'device', devspec)
        device = matching_devices[0]
        return_dict['device_id'] = device['id']
    attached_device_ids = get_attached_dev_ids(volume)
    if target_state == 'present':
        if len(attached_device_ids) == 0:
            do_attach(packet_conn, volume['id'], device['id'])
            return_dict['changed'] = True
        elif device['id'] not in attached_device_ids:
            raise Exception('volume {0} is already attached to device {1}'.format(volume, attached_device_ids))
    elif device is None:
        if len(attached_device_ids) > 0:
            do_detach(packet_conn, volume)
            return_dict['changed'] = True
    elif device['id'] in attached_device_ids:
        do_detach(packet_conn, volume, device['id'])
        return_dict['changed'] = True
    return return_dict