from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ObjectModule(OpenStackModule):
    argument_spec = dict(container=dict(required=True), data=dict(), filename=dict(), name=dict(required=True), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(mutually_exclusive=[('data', 'filename')], required_if=[('state', 'present', ('data', 'filename'), True)], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        object = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, object))
        if state == 'present' and (not object):
            object = self._create()
            self.exit_json(changed=True, object=dict(metadata=object.metadata, **object.to_dict(computed=False)))
        elif state == 'present' and object:
            update = self._build_update(object)
            if update:
                object = self._update(object, update)
            self.exit_json(changed=bool(update), object=dict(metadata=object.metadata, **object.to_dict(computed=False)))
        elif state == 'absent' and object:
            self._delete(object)
            self.exit_json(changed=True)
        elif state == 'absent' and (not object):
            self.exit_json(changed=False)

    def _build_update(self, object):
        update = {}
        container_name = self.params['container']
        filename = self.params['filename']
        if filename is not None:
            if self.conn.object_store.is_object_stale(container_name, object.id, filename):
                update['filename'] = filename
        return update

    def _create(self):
        name = self.params['name']
        container_name = self.params['container']
        kwargs = dict(((k, self.params[k]) for k in ['data', 'filename'] if self.params[k] is not None))
        return self.conn.object_store.create_object(container_name, name, **kwargs)

    def _delete(self, object):
        container_name = self.params['container']
        self.conn.object_store.delete_object(object.id, container=container_name)

    def _find(self):
        name_or_id = self.params['name']
        container_name = self.params['container']
        try:
            return self.conn.object_store.get_object(name_or_id, container=container_name)
        except self.sdk.exceptions.ResourceNotFound:
            return None

    def _update(self, object, update):
        filename = update.get('filename')
        if filename:
            container_name = self.params['container']
            object = self.conn.object_store.create_object(container_name, object.id, filename=filename)
        return object

    def _will_change(self, state, object):
        if state == 'present' and (not object):
            return True
        elif state == 'present' and object:
            return bool(self._build_update(object))
        elif state == 'absent' and object:
            return True
        else:
            return False