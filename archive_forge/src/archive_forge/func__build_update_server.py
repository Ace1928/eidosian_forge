from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
import copy
def _build_update_server(self, server):
    update = {}
    required_metadata = self._parse_metadata(self.params['metadata'])
    assigned_metadata = server.metadata
    add_metadata = dict()
    for k, v in required_metadata.items():
        if k not in assigned_metadata or assigned_metadata[k] != v:
            add_metadata[k] = v
    if add_metadata:
        update['add_metadata'] = add_metadata
    remove_metadata = dict()
    for k, v in assigned_metadata.items():
        if k not in required_metadata or required_metadata[k] != v:
            remove_metadata[k] = v
    if remove_metadata:
        update['remove_metadata'] = remove_metadata
    server_attributes = dict(((k, self.params[k]) for k in ['access_ipv4', 'access_ipv6', 'hostname', 'disk_config', 'description'] if k in self.params and self.params[k] is not None and (self.params[k] != server[k])))
    if server_attributes:
        update['server_attributes'] = server_attributes
    return update