from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_update_type_encryption(self, type_encryption):
    attributes_map = {'encryption_provider': 'provider', 'encryption_cipher': 'cipher', 'encryption_key_size': 'key_size', 'encryption_control_location': 'control_location'}
    encryption_attributes = {attributes_map[k]: self.params[k] for k in self.params if k in attributes_map.keys() and self.params.get(k) is not None and (self.params.get(k) != type_encryption.get(attributes_map[k]))}
    if 'encryption_provider' in encryption_attributes.keys():
        encryption_attributes['provider'] = encryption_attributes['encryption_provider']
    return encryption_attributes