from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def upgrade_hardware(client_obj, array_name_or_serial):
    if utils.is_null_or_empty(array_name_or_serial):
        return (False, False, 'Hardware update failed as no array name is provided.', {}, {})
    try:
        fc_config_resp = client_obj.fibre_channel_configs.get(id=None, group_leader_array=array_name_or_serial)
        if fc_config_resp is None:
            return (False, False, f"No fibre channel config is present for array '{array_name_or_serial}'.", {}, {})
        else:
            fc_config_resp = client_obj.fibre_channel_configs.hw_upgrade(fc_config_resp.attrs.get('id'))
            if hasattr(fc_config_resp, 'attrs'):
                fc_config_resp = fc_config_resp.attrs
            return (True, True, f"Hardware update for group leader array '{array_name_or_serial}' done successfully", {}, fc_config_resp)
    except Exception as ex:
        return (False, False, f"Hardware update failed |'{ex}'", {}, {})