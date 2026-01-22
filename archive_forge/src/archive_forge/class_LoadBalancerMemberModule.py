from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class LoadBalancerMemberModule(OpenStackModule):
    argument_spec = dict(address=dict(), monitor_address=dict(), monitor_port=dict(type='int'), name=dict(required=True), pool=dict(required=True), protocol_port=dict(type='int'), state=dict(default='present', choices=['absent', 'present']), subnet_id=dict(), weight=dict(type='int'))
    module_kwargs = dict(required_if=[('state', 'present', ('address', 'protocol_port'))], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        member, pool = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, member, pool))
        if state == 'present' and (not member):
            member = self._create(pool)
            self.exit_json(changed=True, member=member.to_dict(computed=False), pool=pool.to_dict(computed=False))
        elif state == 'present' and member:
            update = self._build_update(member, pool)
            if update:
                member = self._update(member, pool, update)
            self.exit_json(changed=bool(update), member=member.to_dict(computed=False), pool=pool.to_dict(computed=False))
        elif state == 'absent' and member:
            self._delete(member, pool)
            self.exit_json(changed=True)
        elif state == 'absent' and (not member):
            self.exit_json(changed=False)

    def _build_update(self, member, pool):
        update = {}
        non_updateable_keys = [k for k in ['address', 'name', 'protocol_port', 'subnet_id'] if self.params[k] is not None and self.params[k] != member[k]]
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['monitor_address', 'monitor_port', 'weight'] if self.params[k] is not None and self.params[k] != member[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self, pool):
        kwargs = dict(((k, self.params[k]) for k in ['address', 'monitor_address', 'monitor_port', 'name', 'protocol_port', 'subnet_id', 'weight'] if self.params[k] is not None))
        member = self.conn.load_balancer.create_member(pool.id, **kwargs)
        if self.params['wait']:
            member = self.sdk.resource.wait_for_status(self.conn.load_balancer, member, status='active', failures=['error'], wait=self.params['timeout'], attribute='provisioning_status')
        return member

    def _delete(self, member, pool):
        self.conn.load_balancer.delete_member(member.id, pool.id)
        if self.params['wait']:
            for count in self.sdk.utils.iterate_timeout(timeout=self.params['timeout'], message='Timeout waiting for load-balancer member to be absent'):
                if self.conn.load_balancer.find_member(member.id, pool.id) is None:
                    break

    def _find(self):
        name = self.params['name']
        pool_name_or_id = self.params['pool']
        pool = self.conn.load_balancer.find_pool(name_or_id=pool_name_or_id, ignore_missing=False)
        member = self.conn.load_balancer.find_member(name, pool.id)
        return (member, pool)

    def _update(self, member, pool, update):
        attributes = update.get('attributes')
        if attributes:
            member = self.conn.load_balancer.update_member(member.id, pool.id, **attributes)
        if self.params['wait']:
            member = self.sdk.resource.wait_for_status(self.conn.load_balancer, member, status='active', failures=['error'], wait=self.params['timeout'], attribute='provisioning_status')
        return member

    def _will_change(self, state, member, pool):
        if state == 'present' and (not member):
            return True
        elif state == 'present' and member:
            return bool(self._build_update(member, pool))
        elif state == 'absent' and member:
            return True
        else:
            return False