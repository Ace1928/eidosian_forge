from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeTypeModule(OpenStackModule):
    argument_spec = dict(name=dict(type='str', required=True), description=dict(type='str', required=False), extra_specs=dict(type='dict', required=False), is_public=dict(type='bool'), state=dict(type='str', default='present', choices=['absent', 'present']))
    module_kwargs = dict(required_if=[('state', 'present', ['is_public'])], supports_check_mode=True)

    @staticmethod
    def _extract_result(details):
        if details is not None:
            return details.to_dict(computed=False)
        return {}

    def run(self):
        state = self.params['state']
        name_or_id = self.params['name']
        volume_type = self.conn.block_storage.find_type(name_or_id)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, volume_type))
        if state == 'present' and (not volume_type):
            create_result = self._create()
            volume_type = self._extract_result(create_result)
            self.exit_json(changed=True, volume_type=volume_type)
        elif state == 'present' and volume_type:
            update = self._build_update(volume_type)
            update_result = self._update(volume_type, update)
            volume_type = self._extract_result(update_result)
            self.exit_json(changed=bool(update), volume_type=volume_type)
        elif state == 'absent' and volume_type:
            self._delete(volume_type)
            self.exit_json(changed=True)

    def _build_update(self, volume_type):
        return {**self._build_update_extra_specs(volume_type), **self._build_update_volume_type(volume_type)}

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

    def _build_update_volume_type(self, volume_type):
        update = {}
        allowed_attributes = ['is_public', 'description', 'name']
        type_attributes = {k: self.params[k] for k in allowed_attributes if k in self.params and self.params.get(k) is not None and (self.params.get(k) != volume_type.get(k))}
        if type_attributes:
            update['type_attributes'] = type_attributes
        return update

    def _create(self):
        kwargs = {k: self.params[k] for k in ['name', 'is_public', 'description', 'extra_specs'] if self.params.get(k) is not None}
        volume_type = self.conn.block_storage.create_type(**kwargs)
        return volume_type

    def _delete(self, volume_type):
        self.conn.block_storage.delete_type(volume_type.id)

    def _update(self, volume_type, update):
        if not update:
            return volume_type
        volume_type = self._update_volume_type(volume_type, update)
        volume_type = self._update_extra_specs(volume_type, update)
        return volume_type

    def _update_extra_specs(self, volume_type, update):
        delete_extra_specs_keys = update.get('delete_extra_specs_keys')
        if delete_extra_specs_keys:
            self.conn.block_storage.delete_type_extra_specs(volume_type, delete_extra_specs_keys)
            volume_type = self.conn.block_storage.find_type(volume_type.id)
        create_extra_specs = update.get('create_extra_specs')
        if create_extra_specs:
            self.conn.block_storage.update_type_extra_specs(volume_type, **create_extra_specs)
            volume_type = self.conn.block_storage.find_type(volume_type.id)
        return volume_type

    def _update_volume_type(self, volume_type, update):
        type_attributes = update.get('type_attributes')
        if type_attributes:
            updated_type = self.conn.block_storage.update_type(volume_type, **type_attributes)
            return updated_type
        return volume_type

    def _will_change(self, state, volume_type):
        if state == 'present' and (not volume_type):
            return True
        if state == 'present' and volume_type:
            return bool(self._build_update(volume_type))
        if state == 'absent' and volume_type:
            return True
        return False