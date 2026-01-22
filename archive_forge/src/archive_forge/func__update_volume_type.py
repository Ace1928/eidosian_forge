from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _update_volume_type(self, volume_type, update):
    type_attributes = update.get('type_attributes')
    if type_attributes:
        updated_type = self.conn.block_storage.update_type(volume_type, **type_attributes)
        return updated_type
    return volume_type