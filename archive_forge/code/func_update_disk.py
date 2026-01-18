from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_disk(client_obj, slot, shelf_location, **kwargs):
    if utils.is_null_or_empty(shelf_location):
        return (False, False, 'Disk update failed as no shelf location provided.', {}, {})
    try:
        disk_resp = client_obj.disks.list(detail=True)
        if disk_resp is None:
            return (False, False, 'No Disk is present on array.', {}, {})
        else:
            disk_id = None
            changed_attrs_dict = {}
            for disk_obj in disk_resp:
                if slot == disk_obj.attrs.get('slot') and shelf_location == disk_obj.attrs.get('shelf_location'):
                    disk_id = disk_obj.attrs.get('id')
                    break
            params = utils.remove_null_args(**kwargs)
            disk_resp = client_obj.disks.update(id=disk_id, **params)
            if hasattr(disk_resp, 'attrs'):
                disk_resp = disk_resp.attrs
            changed_attrs_dict['slot'] = slot
            changed_attrs_dict['shelf_location'] = shelf_location
            return (True, True, f"Successfully updated disk to slot '{slot}' at shelf location '{shelf_location}'.", changed_attrs_dict, disk_resp)
    except Exception as ex:
        return (False, False, f"Disk update failed |'{ex}'", {}, {})