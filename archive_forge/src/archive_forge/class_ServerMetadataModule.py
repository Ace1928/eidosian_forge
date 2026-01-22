from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ServerMetadataModule(OpenStackModule):
    argument_spec = dict(name=dict(required=True, aliases=['server']), metadata=dict(required=True, type='dict', aliases=['meta']), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        state = self.params['state']
        server_name_or_id = self.params['name']
        metadata = self.params['metadata']
        server = self.conn.compute.find_server(server_name_or_id, ignore_missing=False)
        server = self.conn.compute.get_server(server.id)
        if self.ansible.check_mode:
            self.exit_json(**self._check_mode_values(state, server, metadata))
        changed = False
        if state == 'present':
            update = self._build_update(server.metadata, metadata)
            if update:
                new_metadata = server.metadata or {}
                new_metadata.update(update)
                self.conn.compute.set_server_metadata(server, **new_metadata)
                changed = True
        elif state == 'absent':
            keys_to_delete = self._get_keys_to_delete(server.metadata, metadata)
            if keys_to_delete:
                self.conn.compute.delete_server_metadata(server, keys_to_delete)
                changed = True
        self.exit_json(changed=changed, server=server.to_dict(computed=False))

    def _build_update(self, current=None, requested=None):
        current = current or {}
        requested = requested or {}
        update = dict(requested.items() - current.items())
        return update

    def _get_keys_to_delete(self, current=None, requested=None):
        current = current or {}
        requested = requested or {}
        return set(current.keys() & requested.keys())

    def _check_mode_values(self, state, server, meta):
        """Builds return values for check mode"""
        changed = False
        if state == 'present':
            update = self._build_update(server.metadata, meta)
            if update:
                changed = True
                new_metadata = server.metadata or {}
                new_metadata.update(update)
                server.metadata = new_metadata
        else:
            keys = self._get_keys_to_delete(server.metadata, meta)
            for k in keys:
                server.meta.pop(k)
        return dict(changed=changed, server=server.to_dict(computed=False))