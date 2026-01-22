from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _update_type_encryption(self, type_encryption, update):
    if update:
        updated_type = self.conn.block_storage.update_type_encryption(encryption=type_encryption, **update)
        return updated_type
    return {}