from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
import copy
class ServerModule(OpenStackModule):
    argument_spec = dict(auto_ip=dict(default=True, type='bool', aliases=['auto_floating_ip', 'public_ip']), availability_zone=dict(), boot_from_volume=dict(default=False, type='bool'), boot_volume=dict(aliases=['root_volume']), config_drive=dict(default=False, type='bool'), delete_ips=dict(default=False, type='bool', aliases=['delete_fip']), description=dict(), flavor=dict(), flavor_include=dict(), flavor_ram=dict(type='int'), floating_ip_pools=dict(type='list', elements='str'), floating_ips=dict(type='list', elements='str'), image=dict(), image_exclude=dict(default='(deprecated)'), key_name=dict(), metadata=dict(type='raw', aliases=['meta']), name=dict(required=True), network=dict(), nics=dict(default=[], type='list', elements='raw'), reuse_ips=dict(default=True, type='bool'), scheduler_hints=dict(type='dict'), security_groups=dict(default=[], type='list', elements='str'), state=dict(default='present', choices=['absent', 'present']), terminate_volume=dict(default=False, type='bool'), userdata=dict(), volume_size=dict(type='int'), volumes=dict(default=[], type='list', elements='str'))
    module_kwargs = dict(mutually_exclusive=[['auto_ip', 'floating_ips', 'floating_ip_pools'], ['flavor', 'flavor_ram'], ['image', 'boot_volume'], ['boot_from_volume', 'boot_volume'], ['nics', 'network']], required_if=[('boot_from_volume', True, ['volume_size', 'image']), ('state', 'present', ('image', 'boot_volume'), True), ('state', 'present', ('flavor', 'flavor_ram'), True)], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        server = self.conn.compute.find_server(self.params['name'])
        if server:
            server = self.conn.compute.get_server(server)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, server))
        if state == 'present' and (not server):
            server = self._create()
            self.exit_json(changed=True, server=server.to_dict(computed=False))
        elif state == 'present' and server:
            update = self._build_update(server)
            if update:
                server = self._update(server, update)
            self.exit_json(changed=bool(update), server=server.to_dict(computed=False))
        elif state == 'absent' and server:
            self._delete(server)
            self.exit_json(changed=True)
        elif state == 'absent' and (not server):
            self.exit_json(changed=False)

    def _build_update(self, server):
        if server.status not in ('ACTIVE', 'SHUTOFF', 'PAUSED', 'SUSPENDED'):
            self.fail_json(msg='The instance is available but not active state: {0}'.format(server.status))
        return {**self._build_update_ips(server), **self._build_update_security_groups(server), **self._build_update_server(server)}

    def _build_update_ips(self, server):
        auto_ip = self.params['auto_ip']
        floating_ips = self.params['floating_ips']
        floating_ip_pools = self.params['floating_ip_pools']
        if not (auto_ip or floating_ips or floating_ip_pools):
            return {}
        ips = [interface_spec['addr'] for v in server['addresses'].values() for interface_spec in v if interface_spec.get('OS-EXT-IPS:type', None) == 'floating']
        if auto_ip and ips and (not floating_ip_pools) and (not floating_ips):
            return {}
        if not ips:
            return dict(ips=dict(auto_ip=auto_ip, ips=floating_ips, ip_pool=floating_ip_pools))
        if auto_ip or not floating_ips:
            return {}
        update = {}
        add_ips = [ip for ip in floating_ips if ip not in ips]
        if add_ips:
            update['add_ips'] = add_ips
        remove_ips = [ip for ip in ips if ip not in floating_ips]
        if remove_ips:
            update['remove_ips'] = remove_ips

    def _build_update_security_groups(self, server):
        update = {}
        required_security_groups = dict(((sg['id'], sg) for sg in [self.conn.network.find_security_group(security_group_name_or_id, ignore_missing=False) for security_group_name_or_id in self.params['security_groups']]))
        server = self.conn.compute.fetch_server_security_groups(server)
        assigned_security_groups = dict(((sg['id'], self.conn.network.get_security_group(sg['id'])) for sg in server.security_groups))
        add_security_groups = [sg for sg_id, sg in required_security_groups.items() if sg_id not in assigned_security_groups]
        if add_security_groups:
            update['add_security_groups'] = add_security_groups
        remove_security_groups = [sg for sg_id, sg in assigned_security_groups.items() if sg_id not in required_security_groups]
        if remove_security_groups:
            update['remove_security_groups'] = remove_security_groups
        return update

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

    def _create(self):
        for k in ['auto_ip', 'floating_ips', 'floating_ip_pools']:
            if self.params[k] is not None and self.params['wait'] is False:
                self.fail_json(msg="Option '{0}' requires 'wait: true'".format(k))
        flavor_name_or_id = self.params['flavor']
        image_id = None
        if not self.params['boot_volume']:
            image_id = self.conn.get_image_id(self.params['image'], self.params['image_exclude'])
            if not image_id:
                self.fail_json(msg='Could not find image {0} with exclude {1}'.format(self.params['image'], self.params['image_exclude']))
        if flavor_name_or_id:
            flavor = self.conn.compute.find_flavor(flavor_name_or_id, ignore_missing=False)
        else:
            flavor = self.conn.get_flavor_by_ram(self.params['flavor_ram'], self.params['flavor_include'])
            if not flavor:
                self.fail_json(msg='Could not find any matching flavor')
        args = dict(flavor=flavor.id, image=image_id, ip_pool=self.params['floating_ip_pools'], ips=self.params['floating_ips'], meta=self._parse_metadata(self.params['metadata']), nics=self._parse_nics())
        for k in ['auto_ip', 'availability_zone', 'boot_from_volume', 'boot_volume', 'config_drive', 'description', 'key_name', 'name', 'network', 'reuse_ips', 'scheduler_hints', 'security_groups', 'terminate_volume', 'timeout', 'userdata', 'volume_size', 'volumes', 'wait']:
            if self.params[k] is not None:
                args[k] = self.params[k]
        server = self.conn.create_server(**args)
        return self.conn.compute.get_server(server)

    def _delete(self, server):
        self.conn.delete_server(server.id, **dict(((k, self.params[k]) for k in ['wait', 'timeout', 'delete_ips'])))

    def _update(self, server, update):
        server = self._update_ips(server, update)
        server = self._update_security_groups(server, update)
        server = self._update_server(server, update)
        return self.conn.compute.get_server(server)

    def _update_ips(self, server, update):
        args = dict(((k, self.params[k]) for k in ['wait', 'timeout']))
        ips = update.get('ips')
        if ips:
            server = self.conn.add_ips_to_server(server, **ips, **args)
        add_ips = update.get('add_ips')
        if add_ips:
            server = self.conn.add_ip_list(server, add_ips, **args)
        remove_ips = update.get('remove_ips')
        if remove_ips:
            for ip in remove_ips:
                ip_id = self.conn.network.find_ip(name_or_id=ip, ignore_missing=False).id
                self.conn.detach_ip_from_server(server_id=server.id, floating_ip_id=ip_id)
        return server

    def _update_security_groups(self, server, update):
        add_security_groups = update.get('add_security_groups')
        if add_security_groups:
            for sg in add_security_groups:
                self.conn.compute.add_security_group_to_server(server, sg)
        remove_security_groups = update.get('remove_security_groups')
        if remove_security_groups:
            for sg in remove_security_groups:
                self.conn.compute.remove_security_group_from_server(server, sg)
        return server

    def _update_server(self, server, update):
        add_metadata = update.get('add_metadata')
        if add_metadata:
            self.conn.compute.set_server_metadata(server.id, **add_metadata)
        remove_metadata = update.get('remove_metadata')
        if remove_metadata:
            self.conn.compute.delete_server_metadata(server.id, remove_metadata.keys())
        server_attributes = update.get('server_attributes')
        if server_attributes:
            server = self.conn.compute.update_server(server['id'], **server_attributes)
        return server

    def _parse_metadata(self, metadata):
        if not metadata:
            return {}
        if isinstance(metadata, str):
            metas = {}
            for kv_str in metadata.split(','):
                k, v = kv_str.split('=')
                metas[k] = v
            return metas
        return metadata

    def _parse_nics(self):
        nics = []
        stringified_nets = self.params['nics']
        if not isinstance(stringified_nets, list):
            self.fail_json(msg="The 'nics' parameter must be a list.")
        nets = [(dict((nested_net.split('='),)) for nested_net in net.split(',')) if isinstance(net, str) else net for net in stringified_nets]
        for net in nets:
            if not isinstance(net, dict):
                self.fail_json(msg="Each entry in the 'nics' parameter must be a dict.")
            if net.get('net-id'):
                nics.append(net)
            elif net.get('net-name'):
                network_id = self.conn.network.find_network(net['net-name'], ignore_missing=False).id
                net = copy.deepcopy(net)
                del net['net-name']
                net['net-id'] = network_id
                nics.append(net)
            elif net.get('port-id'):
                nics.append(net)
            elif net.get('port-name'):
                port_id = self.conn.network.find_port(net['port-name'], ignore_missing=False).id
                net = copy.deepcopy(net)
                del net['port-name']
                net['port-id'] = port_id
                nics.append(net)
            if 'tag' in net:
                nics[-1]['tag'] = net['tag']
        return nics

    def _will_change(self, state, server):
        if state == 'present' and (not server):
            return True
        elif state == 'present' and server:
            return bool(self._build_update(server))
        elif state == 'absent' and server:
            return True
        else:
            return False