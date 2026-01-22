from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _will_change(self, state, load_balancer):
    if state == 'present' and (not load_balancer):
        return True
    elif state == 'present' and load_balancer:
        return bool(self._build_update(load_balancer)[0])
    elif state == 'absent' and load_balancer:
        return True
    else:
        return False