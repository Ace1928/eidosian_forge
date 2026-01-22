from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class PortModule(OpenStackModule):
    argument_spec = dict(allowed_address_pairs=dict(type='list', elements='dict'), binding_profile=dict(type='dict'), binding_vnic_type=dict(choices=['normal', 'direct', 'direct-physical', 'macvtap', 'baremetal', 'virtio-forwarder'], aliases=['vnic_type']), description=dict(), device_id=dict(), device_owner=dict(), dns_domain=dict(), dns_name=dict(), extra_dhcp_opts=dict(type='list', elements='dict'), fixed_ips=dict(type='list', elements='dict'), is_admin_state_up=dict(type='bool', aliases=['admin_state_up']), mac_address=dict(), name=dict(required=True), network=dict(), no_security_groups=dict(default=False, type='bool'), is_port_security_enabled=dict(type='bool', aliases=['port_security_enabled']), security_groups=dict(type='list', elements='str'), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(mutually_exclusive=[['no_security_groups', 'security_groups']], required_if=[('state', 'present', ('network',))], supports_check_mode=True)

    def run(self):
        network_name_or_id = self.params['network']
        port_name_or_id = self.params['name']
        state = self.params['state']
        network = None
        if network_name_or_id:
            network = self.conn.network.find_network(network_name_or_id, ignore_missing=False)
        port = self.conn.network.find_port(port_name_or_id, **dict(network_id=network.id) if network else dict())
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(network, port, state))
        if state == 'present' and (not port):
            port = self._create(network)
            self.exit_json(changed=True, port=port.to_dict(computed=False))
        elif state == 'present' and port:
            update = self._build_update(port)
            if update:
                port = self._update(port, update)
            self.exit_json(changed=bool(update), port=port.to_dict(computed=False))
        elif state == 'absent' and port:
            self._delete(port)
            self.exit_json(changed=True)
        elif state == 'absent' and (not port):
            self.exit_json(changed=False)

    def _build_update(self, port):
        update = {}
        port_attributes = dict(((k, self.params[k]) for k in ['binding_host_id', 'binding_vnic_type', 'data_plane_status', 'description', 'device_id', 'device_owner', 'is_admin_state_up', 'is_port_security_enabled', 'mac_address', 'numa_affinity_policy'] if k in self.params and self.params[k] is not None and (self.params[k] != port[k])))
        for k in ['binding_profile']:
            if self.params[k] is None:
                continue
            if (self.params[k] or port[k]) and self.params[k] != port[k]:
                port_attributes[k] = self.params[k]
        for k in ['allowed_address_pairs', 'extra_dhcp_opts', 'fixed_ips']:
            if self.params[k] is None:
                continue
            if (self.params[k] or port[k]) and self.params[k] != port[k]:
                port_attributes[k] = self.params[k]
        if self.params['no_security_groups']:
            security_group_ids = []
        elif self.params['security_groups'] is not None:
            security_group_ids = [self.conn.network.find_security_group(security_group_name_or_id, ignore_missing=False).id for security_group_name_or_id in self.params['security_groups']]
        else:
            security_group_ids = None
        if security_group_ids is not None and set(security_group_ids) != set(port['security_group_ids']):
            port_attributes['security_group_ids'] = security_group_ids
        if self.conn.has_service('dns') and self.conn.network.find_extension('dns-integration'):
            port_attributes.update(dict(((k, self.params[k]) for k in ['dns_name', 'dns_domain'] if self.params[k] is not None and self.params[k] != port[k])))
        if port_attributes:
            update['port_attributes'] = port_attributes
        return update

    def _create(self, network):
        args = {}
        args['network_id'] = network.id
        if self.params['no_security_groups']:
            args['security_group_ids'] = []
        elif self.params['security_groups'] is not None:
            args['security_group_ids'] = [self.conn.network.find_security_group(security_group_name_or_id, ignore_missing=False).id for security_group_name_or_id in self.params['security_groups']]
        for k in ['allowed_address_pairs', 'binding_profile', 'binding_vnic_type', 'device_id', 'device_owner', 'description', 'extra_dhcp_opts', 'is_admin_state_up', 'mac_address', 'is_port_security_enabled', 'fixed_ips', 'name']:
            if self.params[k] is not None:
                args[k] = self.params[k]
        if self.conn.has_service('dns') and self.conn.network.find_extension('dns-integration'):
            for k in ['dns_domain', 'dns_name']:
                if self.params[k] is not None:
                    args[k] = self.params[k]
        return self.conn.network.create_port(**args)

    def _delete(self, port):
        self.conn.network.delete_port(port.id)

    def _update(self, port, update):
        port_attributes = update.get('port_attributes')
        if port_attributes:
            port = self.conn.network.update_port(port, **port_attributes)
        return port

    def _will_change(self, port, state):
        if state == 'present' and (not port):
            return True
        elif state == 'present' and port:
            return bool(self._build_update(port))
        elif state == 'absent' and port:
            return True
        else:
            return False