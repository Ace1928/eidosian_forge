from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _add_extra_attrs(self, params):
    duplicates = set(self.params['extra_attrs']) & set(params)
    if duplicates:
        self.fail_json(msg='Duplicate key(s) {0} in extra_specs'.format(list(duplicates)))
    params.update(self.params['extra_attrs'])
    return params