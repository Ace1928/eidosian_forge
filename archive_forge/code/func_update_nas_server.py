from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def update_nas_server(self, nas_server_obj, new_name=None, default_unix_user=None, default_windows_user=None, is_rep_dest=None, is_multiprotocol_enabled=None, allow_unmapped_user=None, is_backup_only=None, is_packet_reflect_enabled=None, current_uds=None, enable_win_to_unix_name_map=None):
    """
        The Details of the NAS Server will be updated in the function.
        """
    try:
        nas_server_obj.modify(name=new_name, is_replication_destination=is_rep_dest, is_backup_only=is_backup_only, is_multi_protocol_enabled=is_multiprotocol_enabled, default_unix_user=default_unix_user, default_windows_user=default_windows_user, allow_unmapped_user=allow_unmapped_user, is_packet_reflect_enabled=is_packet_reflect_enabled, enable_windows_to_unix_username=enable_win_to_unix_name_map, current_unix_directory_service=current_uds)
    except Exception as e:
        error_msg = 'Failed to Update parameters of NAS Server %s with error %s' % (nas_server_obj.name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)