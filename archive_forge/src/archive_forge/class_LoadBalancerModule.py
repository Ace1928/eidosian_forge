from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class LoadBalancerModule(OpenStackModule):
    argument_spec = dict(assign_floating_ip=dict(default=False, type='bool', aliases=['auto_public_ip']), delete_floating_ip=dict(default=False, type='bool', aliases=['delete_public_ip']), description=dict(), flavor=dict(), floating_ip_address=dict(aliases=['public_ip_address']), floating_ip_network=dict(aliases=['public_network']), name=dict(required=True), state=dict(default='present', choices=['absent', 'present']), vip_address=dict(), vip_network=dict(), vip_port=dict(), vip_subnet=dict())
    module_kwargs = dict(required_if=[('state', 'present', ('vip_network', 'vip_subnet', 'vip_port'), True)], mutually_exclusive=[('assign_floating_ip', 'delete_floating_ip')], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        load_balancer = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, load_balancer))
        if state == 'present' and (not load_balancer):
            load_balancer, floating_ip = self._create()
            self.exit_json(changed=True, load_balancer=load_balancer.to_dict(computed=False), **dict(floating_ip=floating_ip.to_dict(computed=False)) if floating_ip is not None else dict())
        elif state == 'present' and load_balancer:
            update, floating_ip = self._build_update(load_balancer)
            if update:
                load_balancer, floating_ip = self._update(load_balancer, update)
            self.exit_json(changed=bool(update), load_balancer=load_balancer.to_dict(computed=False), **dict(floating_ip=floating_ip.to_dict(computed=False)) if floating_ip is not None else dict())
        elif state == 'absent' and load_balancer:
            self._delete(load_balancer)
            self.exit_json(changed=True)
        elif state == 'absent' and (not load_balancer):
            self.exit_json(changed=False)

    def _build_update(self, load_balancer):
        update = {}
        non_updateable_keys = [k for k in ['name', 'vip_address'] if self.params[k] is not None and self.params[k] != load_balancer[k]]
        flavor_name_or_id = self.params['flavor']
        if flavor_name_or_id is not None:
            flavor = self.conn.load_balancer.find_flavor(flavor_name_or_id, ignore_missing=False)
            if load_balancer['flavor_id'] != flavor.id:
                non_updateable_keys.append('flavor_id')
        vip_network_name_or_id = self.params['vip_network']
        if vip_network_name_or_id is not None:
            network = self.conn.network.find_network(vip_network_name_or_id, ignore_missing=False)
            if load_balancer['vip_network_id'] != network.id:
                non_updateable_keys.append('vip_network_id')
        vip_subnet_name_or_id = self.params['vip_subnet']
        if vip_subnet_name_or_id is not None:
            subnet = self.conn.network.find_subnet(vip_subnet_name_or_id, ignore_missing=False)
            if load_balancer['vip_subnet_id'] != subnet.id:
                non_updateable_keys.append('vip_subnet_id')
        vip_port_name_or_id = self.params['vip_port']
        if vip_port_name_or_id is not None:
            port = self.conn.network.find_port(vip_port_name_or_id, ignore_missing=False)
            if load_balancer['vip_port_id'] != port.id:
                non_updateable_keys.append('vip_port_id')
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['description'] if self.params[k] is not None and self.params[k] != load_balancer[k]))
        if attributes:
            update['attributes'] = attributes
        floating_ip, floating_ip_update = self._build_update_floating_ip(load_balancer)
        return ({**update, **floating_ip_update}, floating_ip)

    def _build_update_floating_ip(self, load_balancer):
        assign_floating_ip = self.params['assign_floating_ip']
        delete_floating_ip = self.params['delete_floating_ip']
        floating_ip_address = self.params['floating_ip_address']
        if floating_ip_address is not None and (not assign_floating_ip and (not delete_floating_ip)):
            self.fail_json(msg='assign_floating_ip or delete_floating_ip must be true when floating_ip_address is set')
        floating_ip_network = self.params['floating_ip_network']
        if floating_ip_network is not None and (not assign_floating_ip and (not delete_floating_ip)):
            self.fail_json(msg='assign_floating_ip or delete_floating_ip must be true when floating_ip_network is set')
        ips = list(self.conn.network.ips(port_id=load_balancer.vip_port_id, fixed_ip_address=load_balancer.vip_address))
        if len(ips) > 1:
            self.fail_json(msg='Only a single floating ip address per load-balancer is supported')
        if delete_floating_ip or not assign_floating_ip:
            if not ips:
                return (None, {})
            if len(ips) != 1:
                raise AssertionError('A single floating ip is expected')
            ip = ips[0]
            return (ip, {'delete_floating_ip': ip})
        if not ips:
            return (None, dict(assign_floating_ip=dict(floating_ip_address=floating_ip_address, floating_ip_network=floating_ip_network)))
        if len(ips) != 1:
            raise AssertionError('A single floating ip is expected')
        ip = ips[0]
        if floating_ip_network is not None:
            network = self.conn.network.find_network(floating_ip_network, ignore_missing=False)
            if ip.floating_network_id != network.id:
                return (ip, dict(assign_floating_ip=dict(floating_ip_address=floating_ip_address, floating_ip_network=floating_ip_network), delete_floating_ip=ip))
        if floating_ip_address is not None and floating_ip_address != ip.floating_ip_address:
            return (ip, dict(assign_floating_ip=dict(floating_ip_address=floating_ip_address, floating_ip_network=floating_ip_network), delete_floating_ip=ip))
        return (ip, {})

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['description', 'name', 'vip_address'] if self.params[k] is not None))
        flavor_name_or_id = self.params['flavor']
        if flavor_name_or_id is not None:
            flavor = self.conn.load_balancer.find_flavor(flavor_name_or_id, ignore_missing=False)
            kwargs['flavor_id'] = flavor.id
        vip_network_name_or_id = self.params['vip_network']
        if vip_network_name_or_id is not None:
            network = self.conn.network.find_network(vip_network_name_or_id, ignore_missing=False)
            kwargs['vip_network_id'] = network.id
        vip_subnet_name_or_id = self.params['vip_subnet']
        if vip_subnet_name_or_id is not None:
            subnet = self.conn.network.find_subnet(vip_subnet_name_or_id, ignore_missing=False)
            kwargs['vip_subnet_id'] = subnet.id
        vip_port_name_or_id = self.params['vip_port']
        if vip_port_name_or_id is not None:
            port = self.conn.network.find_port(vip_port_name_or_id, ignore_missing=False)
            kwargs['vip_port_id'] = port.id
        load_balancer = self.conn.load_balancer.create_load_balancer(**kwargs)
        if self.params['wait']:
            load_balancer = self.conn.load_balancer.wait_for_load_balancer(load_balancer.id, wait=self.params['timeout'])
        floating_ip, update = self._build_update_floating_ip(load_balancer)
        if update:
            load_balancer, floating_ip = self._update_floating_ip(load_balancer, update)
        return (load_balancer, floating_ip)

    def _delete(self, load_balancer):
        if self.params['delete_floating_ip']:
            ips = list(self.conn.network.ips(port_id=load_balancer.vip_port_id, fixed_ip_address=load_balancer.vip_address))
        else:
            ips = []
        self.conn.load_balancer.delete_load_balancer(load_balancer.id, cascade=True)
        if self.params['wait']:
            for count in self.sdk.utils.iterate_timeout(timeout=self.params['timeout'], message='Timeout waiting for load-balancer to be absent'):
                if self.conn.load_balancer.find_load_balancer(load_balancer.id) is None:
                    break
        for ip in ips:
            self.conn.network.delete_ip(ip)

    def _find(self):
        name = self.params['name']
        return self.conn.load_balancer.find_load_balancer(name_or_id=name)

    def _update(self, load_balancer, update):
        attributes = update.get('attributes')
        if attributes:
            load_balancer = self.conn.load_balancer.update_load_balancer(load_balancer.id, **attributes)
        if self.params['wait']:
            load_balancer = self.conn.load_balancer.wait_for_load_balancer(load_balancer.id, wait=self.params['timeout'])
        load_balancer, floating_ip = self._update_floating_ip(load_balancer, update)
        return (load_balancer, floating_ip)

    def _update_floating_ip(self, load_balancer, update):
        floating_ip = None
        delete_floating_ip = update.get('delete_floating_ip')
        if delete_floating_ip:
            self.conn.network.delete_ip(delete_floating_ip.id)
        assign_floating_ip = update.get('assign_floating_ip')
        if assign_floating_ip:
            floating_ip_address = assign_floating_ip['floating_ip_address']
            floating_ip_network = assign_floating_ip['floating_ip_network']
            if floating_ip_network is not None:
                network = self.conn.network.find_network(floating_ip_network, ignore_missing=False)
            else:
                network = None
            if floating_ip_address is not None:
                kwargs = {'floating_network_id': network.id} if network is not None else {}
                ip = self.conn.network.find_ip(floating_ip_address, **kwargs)
            else:
                ip = None
            if ip:
                if ip['port_id'] is not None:
                    self.fail_json(msg='Floating ip {0} is associated to another fixed ip address {1} already'.format(ip.floating_ip_address, ip.fixed_ip_address))
                floating_ip = self.conn.network.update_ip(ip.id, fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
            elif floating_ip_address:
                kwargs = {'floating_network_id': network.id} if network is not None else {}
                floating_ip = self.conn.network.create_ip(fixed_ip_address=load_balancer.vip_address, floating_ip_address=floating_ip_address, port_id=load_balancer.vip_port_id, **kwargs)
            elif network:
                ips = [ip for ip in self.conn.network.ips(floating_network_id=network.id) if ip['port_id'] is None]
                if ips:
                    ip = ips[0]
                    floating_ip = self.conn.network.update_ip(ip.id, fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
                else:
                    floating_ip = self.conn.network.create_ip(fixed_ip_address=load_balancer.vip_address, floating_network_id=network.id, port_id=load_balancer.vip_port_id)
            else:
                ip = self.conn.network.find_available_ip()
                if ip:
                    floating_ip = self.conn.network.update_ip(ip.id, fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
                else:
                    floating_ip = self.conn.network.create_ip(fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
        return (load_balancer, floating_ip)

    def _will_change(self, state, load_balancer):
        if state == 'present' and (not load_balancer):
            return True
        elif state == 'present' and load_balancer:
            return bool(self._build_update(load_balancer)[0])
        elif state == 'absent' and load_balancer:
            return True
        else:
            return False