from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def update_replication_arg_list(replication, replication_args_list, nas_server_obj):
    """ Update replication arg list
        :param: replication: Dict which has all the replication parameter values
        :param: replication_args_list: the existing list which should be updated
        :param: nas_server_obj: NAS Server object on which replication is to be enabled
        :return: Updated replication_args_list
    """
    if 'destination_sp' in replication and replication['destination_sp']:
        dst_sp_enum = get_sp_enum(replication['destination_sp'])
        replication_args_list['dst_sp'] = dst_sp_enum
    replication_args_list['dst_pool_id'] = replication['destination_pool_id']
    if 'is_backup' in replication and replication['is_backup']:
        replication_args_list['is_backup_only'] = replication['is_backup']
    if replication['replication_type'] == 'local':
        replication_args_list['dst_nas_server_name'] = 'DR_' + nas_server_obj.name
        if 'destination_nas_server_name' in replication and replication['destination_nas_server_name'] is not None:
            replication_args_list['dst_nas_server_name'] = replication['destination_nas_server_name']
    elif replication['destination_nas_server_name'] is None:
        replication_args_list['dst_nas_server_name'] = nas_server_obj.name