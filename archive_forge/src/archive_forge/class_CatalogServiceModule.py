from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class CatalogServiceModule(OpenStackModule):
    argument_spec = dict(description=dict(), is_enabled=dict(aliases=['enabled'], type='bool'), name=dict(required=True), type=dict(required=True, aliases=['service_type']), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        state = self.params['state']
        service = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, service))
        if state == 'present' and (not service):
            service = self._create()
            self.exit_json(changed=True, service=service.to_dict(computed=False))
        elif state == 'present' and service:
            update = self._build_update(service)
            if update:
                service = self._update(service, update)
            self.exit_json(changed=bool(update), service=service.to_dict(computed=False))
        elif state == 'absent' and service:
            self._delete(service)
            self.exit_json(changed=True)
        elif state == 'absent' and (not service):
            self.exit_json(changed=False)

    def _build_update(self, service):
        update = {}
        non_updateable_keys = [k for k in ['name'] if self.params[k] is not None and self.params[k] != service[k]]
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['description', 'is_enabled', 'type'] if self.params[k] is not None and self.params[k] != service[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['description', 'is_enabled', 'name', 'type'] if self.params[k] is not None))
        return self.conn.identity.create_service(**kwargs)

    def _delete(self, service):
        self.conn.identity.delete_service(service.id)

    def _find(self):
        kwargs = dict(((k, self.params[k]) for k in ['name', 'type']))
        matches = list(self.conn.identity.services(**kwargs))
        if len(matches) > 1:
            self.fail_json(msg='Found more a single service matching the given parameters.')
        elif len(matches) == 1:
            return matches[0]
        else:
            return None

    def _update(self, service, update):
        attributes = update.get('attributes')
        if attributes:
            service = self.conn.identity.update_service(service.id, **attributes)
        return service

    def _will_change(self, state, service):
        if state == 'present' and (not service):
            return True
        elif state == 'present' and service:
            return bool(self._build_update(service))
        elif state == 'absent' and service:
            return True
        else:
            return False