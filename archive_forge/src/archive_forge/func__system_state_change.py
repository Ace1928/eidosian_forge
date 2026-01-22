from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _system_state_change(self, state, server_group):
    if state == 'present' and (not server_group):
        return True
    if state == 'absent' and server_group:
        return True
    return False