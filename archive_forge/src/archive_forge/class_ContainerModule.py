from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ContainerModule(OpenStackModule):
    argument_spec = dict(delete_metadata_keys=dict(type='list', elements='str', no_log=False, aliases=['keys']), delete_with_all_objects=dict(type='bool', default=False), metadata=dict(type='dict'), name=dict(required=True, aliases=['container']), read_ACL=dict(), state=dict(default='present', choices=['present', 'absent']), write_ACL=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        state = self.params['state']
        container = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, container))
        if state == 'present' and (not container):
            container = self._create()
            self.exit_json(changed=True, container=dict(metadata=container.metadata, **container.to_dict(computed=False)))
        elif state == 'present' and container:
            update = self._build_update(container)
            if update:
                container = self._update(container, update)
            self.exit_json(changed=bool(update), container=dict(metadata=container.metadata, **container.to_dict(computed=False)))
        elif state == 'absent' and container:
            self._delete(container)
            self.exit_json(changed=True)
        elif state == 'absent' and (not container):
            self.exit_json(changed=False)

    def _build_update(self, container):
        update = {}
        metadata = self.params['metadata']
        if metadata is not None:
            old_metadata = dict(((k.lower(), v) for k, v in container.metadata or {}))
            new_metadata = dict(((k, v) for k, v in metadata.items() if k.lower() not in old_metadata or v != old_metadata[k.lower()]))
            if new_metadata:
                update['metadata'] = new_metadata
        delete_metadata_keys = self.params['delete_metadata_keys']
        if delete_metadata_keys is not None:
            for key in delete_metadata_keys:
                if container.metadata is not None and key.lower() in [k.lower() for k in container.metadata.keys()]:
                    update['delete_metadata_keys'] = delete_metadata_keys
                    break
        attributes = dict(((k, self.params[k]) for k in ['read_ACL', 'write_ACL'] if self.params[k] is not None and self.params[k] != container[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['metadata', 'name', 'read_ACL', 'write_ACL'] if self.params[k] is not None))
        return self.conn.object_store.create_container(**kwargs)

    def _delete(self, container):
        if self.params['delete_with_all_objects']:
            for object in self.conn.object_store.objects(container.name):
                self.conn.object_store.delete_object(obj=object.name, container=container.name)
        self.conn.object_store.delete_container(container=container.name)

    def _find(self):
        name_or_id = self.params['name']
        try:
            return self.conn.object_store.get_container_metadata(name_or_id)
        except self.sdk.exceptions.ResourceNotFound:
            return None

    def _update(self, container, update):
        delete_metadata_keys = update.get('delete_metadata_keys')
        if delete_metadata_keys:
            self.conn.object_store.delete_container_metadata(container=container.name, keys=delete_metadata_keys)
            container = self.conn.object_store.get_container_metadata(container.name)
        metadata = update.get('metadata')
        if metadata:
            container = self.conn.object_store.set_container_metadata(container.name, refresh=True, **metadata)
        attributes = update.get('attributes')
        if attributes:
            container = self.conn.object_store.set_container_metadata(container.name, refresh=True, **attributes)
        return container

    def _will_change(self, state, container):
        if state == 'present' and (not container):
            return True
        elif state == 'present' and container:
            return bool(self._build_update(container))
        elif state == 'absent' and container:
            return True
        else:
            return False