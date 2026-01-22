from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class LoadBalancerPoolModule(OpenStackModule):
    argument_spec = dict(description=dict(), lb_algorithm=dict(default='ROUND_ROBIN'), listener=dict(), loadbalancer=dict(), name=dict(required=True), protocol=dict(default='HTTP'), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(required_if=[('state', 'present', ('listener', 'loadbalancer'), True)], mutually_exclusive=[('listener', 'loadbalancer')], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        pool = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, pool))
        if state == 'present' and (not pool):
            pool = self._create()
            self.exit_json(changed=True, pool=pool.to_dict(computed=False))
        elif state == 'present' and pool:
            update = self._build_update(pool)
            if update:
                pool = self._update(pool, update)
            self.exit_json(changed=bool(update), pool=pool.to_dict(computed=False))
        elif state == 'absent' and pool:
            self._delete(pool)
            self.exit_json(changed=True)
        elif state == 'absent' and (not pool):
            self.exit_json(changed=False)

    def _build_update(self, pool):
        update = {}
        non_updateable_keys = [k for k in ['protocol'] if self.params[k] is not None and self.params[k] != pool[k]]
        listener_name_or_id = self.params['listener']
        if listener_name_or_id:
            listener = self.conn.load_balancer.find_listener(listener_name_or_id, ignore_missing=False)
            if pool['listeners'] != [dict(id=listener.id)]:
                non_updateable_keys.append('listener_id')
        loadbalancer_name_or_id = self.params['loadbalancer']
        if loadbalancer_name_or_id:
            loadbalancer = self.conn.load_balancer.find_load_balancer(loadbalancer_name_or_id, ignore_missing=False)
            if listener['load_balancers'] != [dict(id=loadbalancer.id)]:
                non_updateable_keys.append('loadbalancer_id')
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['description', 'lb_algorithm'] if self.params[k] is not None and self.params[k] != pool[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['description', 'name', 'protocol', 'lb_algorithm'] if self.params[k] is not None))
        listener_name_or_id = self.params['listener']
        if listener_name_or_id:
            listener = self.conn.load_balancer.find_listener(listener_name_or_id, ignore_missing=False)
            kwargs['listener_id'] = listener.id
        loadbalancer_name_or_id = self.params['loadbalancer']
        if loadbalancer_name_or_id:
            loadbalancer = self.conn.load_balancer.find_load_balancer(loadbalancer_name_or_id, ignore_missing=False)
            kwargs['loadbalancer_id'] = loadbalancer.id
        pool = self.conn.load_balancer.create_pool(**kwargs)
        if self.params['wait']:
            pool = self.sdk.resource.wait_for_status(self.conn.load_balancer, pool, status='active', failures=['error'], wait=self.params['timeout'], attribute='provisioning_status')
        return pool

    def _delete(self, pool):
        self.conn.load_balancer.delete_pool(pool.id)
        if self.params['wait']:
            for count in self.sdk.utils.iterate_timeout(timeout=self.params['timeout'], message='Timeout waiting for load-balancer pool to be absent'):
                if self.conn.load_balancer.find_pool(pool.id) is None:
                    break

    def _find(self):
        name = self.params['name']
        return self.conn.load_balancer.find_pool(name_or_id=name)

    def _update(self, pool, update):
        attributes = update.get('attributes')
        if attributes:
            pool = self.conn.load_balancer.update_pool(pool.id, **attributes)
        if self.params['wait']:
            pool = self.sdk.resource.wait_for_status(self.conn.load_balancer, pool, status='active', failures=['error'], wait=self.params['timeout'], attribute='provisioning_status')
        return pool

    def _will_change(self, state, pool):
        if state == 'present' and (not pool):
            return True
        elif state == 'present' and pool:
            return bool(self._build_update(pool))
        elif state == 'absent' and pool:
            return True
        else:
            return False