from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def update_replication_params(self, replication, replication_reuse_resource):
    """ Update replication dict with remote system information
            :param: replication: Dict which has all the replication parameter values
            :return: Updated replication Dict
        """
    try:
        if 'replication_type' in replication and replication['replication_type'] == 'remote':
            connection_params = {'unispherehost': replication['remote_system']['remote_system_host'], 'username': replication['remote_system']['remote_system_username'], 'password': replication['remote_system']['remote_system_password'], 'validate_certs': replication['remote_system']['remote_system_verifycert'], 'port': replication['remote_system']['remote_system_port']}
            remote_system_conn = utils.get_unity_unisphere_connection(connection_params, application_type)
            replication['remote_system_name'] = remote_system_conn.name
            if replication['destination_pool_name'] is not None:
                pool_object = remote_system_conn.get_pool(name=replication['destination_pool_name'])
                replication['destination_pool_id'] = pool_object.id
            if replication['destination_nas_server_name'] is not None and replication_reuse_resource:
                nas_object = remote_system_conn.get_nas_server(name=replication['destination_nas_server_name'])
                replication['destination_nas_server_id'] = nas_object.id
        else:
            replication['remote_system_name'] = self.unity_conn.name
            if replication['destination_pool_name'] is not None:
                pool_object = self.unity_conn.get_pool(name=replication['destination_pool_name'])
                replication['destination_pool_id'] = pool_object.id
    except Exception as e:
        errormsg = 'Updating replication params failed with error %s' % str(e)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)