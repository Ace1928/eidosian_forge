from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_update_security_group(self, security_group):
    update = {}
    non_updateable_keys = [k for k in [] if self.params[k] is not None and self.params[k] != security_group[k]]
    if non_updateable_keys:
        self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
    attributes = dict(((k, self.params[k]) for k in ['description'] if self.params[k] is not None and self.params[k] != security_group[k]))
    if attributes:
        update['attributes'] = attributes
    return update