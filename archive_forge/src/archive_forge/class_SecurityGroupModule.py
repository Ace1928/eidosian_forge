from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class SecurityGroupModule(OpenStackModule):
    argument_spec = dict(description=dict(), name=dict(required=True), project=dict(), security_group_rules=dict(type='list', elements='dict', options=dict(description=dict(), direction=dict(default='ingress', choices=['egress', 'ingress']), ether_type=dict(default='IPv4', choices=['IPv4', 'IPv6']), port_range_max=dict(type='int'), port_range_min=dict(type='int'), protocol=dict(), remote_group=dict(), remote_ip_prefix=dict())), state=dict(default='present', choices=['absent', 'present']), stateful=dict(type='bool'))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        state = self.params['state']
        security_group = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, security_group))
        if state == 'present' and (not security_group):
            security_group = self._create()
            self.exit_json(changed=True, security_group=security_group.to_dict(computed=False))
        elif state == 'present' and security_group:
            update = self._build_update(security_group)
            if update:
                security_group = self._update(security_group, update)
            self.exit_json(changed=bool(update), security_group=security_group.to_dict(computed=False))
        elif state == 'absent' and security_group:
            self._delete(security_group)
            self.exit_json(changed=True)
        elif state == 'absent' and (not security_group):
            self.exit_json(changed=False)

    def _build_update(self, security_group):
        return {**self._build_update_security_group(security_group), **self._build_update_security_group_rules(security_group)}

    def _build_update_security_group(self, security_group):
        update = {}
        non_updateable_keys = [k for k in [] if self.params[k] is not None and self.params[k] != security_group[k]]
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['description'] if self.params[k] is not None and self.params[k] != security_group[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _build_update_security_group_rules(self, security_group):
        if self.params['security_group_rules'] is None:
            return {}

        def find_security_group_rule_match(prototype, security_group_rules):
            matches = [r for r in security_group_rules if is_security_group_rule_match(prototype, r)]
            if len(matches) > 1:
                self.fail_json(msg='Found more a single matching security group rule which match the given parameters.')
            elif len(matches) == 1:
                return matches[0]
            else:
                return None

        def is_security_group_rule_match(prototype, security_group_rule):
            skip_keys = ['ether_type']
            if 'ether_type' in prototype and security_group_rule['ethertype'] != prototype['ether_type']:
                return False
            if 'protocol' in prototype and prototype['protocol'] in ['tcp', 'udp']:
                if 'port_range_max' in prototype and prototype['port_range_max'] in [-1, 65535]:
                    if security_group_rule['port_range_max'] is not None:
                        return False
                    skip_keys.append('port_range_max')
                if 'port_range_min' in prototype and prototype['port_range_min'] in [-1, 1]:
                    if security_group_rule['port_range_min'] is not None:
                        return False
                    skip_keys.append('port_range_min')
            if all((security_group_rule[k] == prototype[k] for k in set(prototype.keys()) - set(skip_keys))):
                return security_group_rule
            else:
                return None
        update = {}
        keep_security_group_rules = {}
        create_security_group_rules = []
        delete_security_group_rules = []
        for prototype in self._generate_security_group_rules(security_group):
            match = find_security_group_rule_match(prototype, security_group.security_group_rules)
            if match:
                keep_security_group_rules[match['id']] = match
            else:
                create_security_group_rules.append(prototype)
        for security_group_rule in security_group.security_group_rules:
            if security_group_rule['id'] not in keep_security_group_rules.keys():
                delete_security_group_rules.append(security_group_rule)
        if create_security_group_rules:
            update['create_security_group_rules'] = create_security_group_rules
        if delete_security_group_rules:
            update['delete_security_group_rules'] = delete_security_group_rules
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['description', 'name', 'stateful'] if self.params[k] is not None))
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(name_or_id=project_name_or_id, ignore_missing=False)
            kwargs['project_id'] = project.id
        security_group = self.conn.network.create_security_group(**kwargs)
        update = self._build_update_security_group_rules(security_group)
        if update:
            security_group = self._update_security_group_rules(security_group, update)
        return security_group

    def _delete(self, security_group):
        self.conn.network.delete_security_group(security_group.id)

    def _find(self):
        kwargs = dict(name_or_id=self.params['name'])
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(name_or_id=project_name_or_id, ignore_missing=False)
            kwargs['project_id'] = project.id
        return self.conn.network.find_security_group(**kwargs)

    def _generate_security_group_rules(self, security_group):
        security_group_cache = {}
        security_group_cache[security_group.name] = security_group
        security_group_cache[security_group.id] = security_group

        def _generate_security_group_rule(params):
            prototype = dict(((k, params[k]) for k in ['description', 'direction', 'remote_ip_prefix'] if params[k] is not None))
            prototype['project_id'] = security_group.project_id
            prototype['security_group_id'] = security_group.id
            remote_group_name_or_id = params['remote_group']
            if remote_group_name_or_id is not None:
                if remote_group_name_or_id in security_group_cache:
                    remote_group = security_group_cache[remote_group_name_or_id]
                else:
                    remote_group = self.conn.network.find_security_group(remote_group_name_or_id, ignore_missing=False)
                    security_group_cache[remote_group_name_or_id] = remote_group
                prototype['remote_group_id'] = remote_group.id
            ether_type = params['ether_type']
            if ether_type is not None:
                prototype['ether_type'] = ether_type
            protocol = params['protocol']
            if protocol is not None and protocol not in ['any', '0']:
                prototype['protocol'] = protocol
            port_range_max = params['port_range_max']
            port_range_min = params['port_range_min']
            if protocol in ['icmp', 'ipv6-icmp']:
                if port_range_max is not None and int(port_range_max) != -1:
                    prototype['port_range_max'] = int(port_range_max)
                if port_range_min is not None and int(port_range_min) != -1:
                    prototype['port_range_min'] = int(port_range_min)
            elif protocol in ['tcp', 'udp']:
                if port_range_max is not None and int(port_range_max) != -1:
                    prototype['port_range_max'] = int(port_range_max)
                if port_range_min is not None and int(port_range_min) != -1:
                    prototype['port_range_min'] = int(port_range_min)
            elif protocol in ['any', '0']:
                pass
            else:
                if port_range_max is not None:
                    prototype['port_range_max'] = int(port_range_max)
                if port_range_min is not None:
                    prototype['port_range_min'] = int(port_range_min)
            return prototype
        return [_generate_security_group_rule(r) for r in self.params['security_group_rules'] or []]

    def _update(self, security_group, update):
        security_group = self._update_security_group(security_group, update)
        return self._update_security_group_rules(security_group, update)

    def _update_security_group(self, security_group, update):
        attributes = update.get('attributes')
        if attributes:
            security_group = self.conn.network.update_security_group(security_group.id, **attributes)
        return security_group

    def _update_security_group_rules(self, security_group, update):
        delete_security_group_rules = update.get('delete_security_group_rules')
        if delete_security_group_rules:
            for security_group_rule in delete_security_group_rules:
                self.conn.network.delete_security_group_rule(security_group_rule['id'])
        create_security_group_rules = update.get('create_security_group_rules')
        if create_security_group_rules:
            self.conn.network.create_security_group_rules(create_security_group_rules)
        if create_security_group_rules or delete_security_group_rules:
            return self.conn.network.get_security_group(security_group.id)
        else:
            return security_group

    def _will_change(self, state, security_group):
        if state == 'present' and (not security_group):
            return True
        elif state == 'present' and security_group:
            return bool(self._build_update(security_group))
        elif state == 'absent' and security_group:
            return True
        else:
            return False