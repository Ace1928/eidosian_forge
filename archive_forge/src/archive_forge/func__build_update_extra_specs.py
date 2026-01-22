from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_update_extra_specs(self, volume_type):
    update = {}
    old_extra_specs = volume_type['extra_specs']
    new_extra_specs = self.params['extra_specs'] or {}
    delete_extra_specs_keys = set(old_extra_specs.keys()) - set(new_extra_specs.keys())
    if delete_extra_specs_keys:
        update['delete_extra_specs_keys'] = delete_extra_specs_keys
    stringified = {k: str(v) for k, v in new_extra_specs.items()}
    if old_extra_specs != stringified:
        update['create_extra_specs'] = new_extra_specs
    return update