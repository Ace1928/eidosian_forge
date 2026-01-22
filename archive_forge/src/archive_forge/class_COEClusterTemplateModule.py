from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class COEClusterTemplateModule(OpenStackModule):
    argument_spec = dict(coe=dict(choices=['kubernetes', 'swarm', 'mesos']), dns_nameserver=dict(), docker_storage_driver=dict(choices=['devicemapper', 'overlay', 'overlay2']), docker_volume_size=dict(type='int'), external_network_id=dict(), fixed_network=dict(), fixed_subnet=dict(), flavor_id=dict(), http_proxy=dict(), https_proxy=dict(), image_id=dict(), is_floating_ip_enabled=dict(type='bool', default=True, aliases=['floating_ip_enabled']), keypair_id=dict(), labels=dict(type='raw'), master_flavor_id=dict(), is_master_lb_enabled=dict(type='bool', default=False, aliases=['master_lb_enabled']), is_public=dict(type='bool', aliases=['public']), is_registry_enabled=dict(type='bool', aliases=['registry_enabled']), is_tls_disabled=dict(type='bool', aliases=['tls_disabled']), name=dict(required=True), network_driver=dict(choices=['flannel', 'calico', 'docker']), no_proxy=dict(), server_type=dict(choices=['vm', 'bm']), state=dict(default='present', choices=['absent', 'present']), volume_driver=dict(choices=['cinder', 'rexray']))
    module_kwargs = dict(required_if=[('state', 'present', ('coe', 'image_id'))], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        cluster_template = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, cluster_template))
        if state == 'present' and (not cluster_template):
            cluster_template = self._create()
            self.exit_json(changed=True, cluster_template=cluster_template.to_dict(computed=False))
        elif state == 'present' and cluster_template:
            update = self._build_update(cluster_template)
            if update:
                cluster_template = self._update(cluster_template, update)
            self.exit_json(changed=bool(update), cluster_template=cluster_template.to_dict(computed=False))
        elif state == 'absent' and cluster_template:
            self._delete(cluster_template)
            self.exit_json(changed=True)
        elif state == 'absent' and (not cluster_template):
            self.exit_json(changed=False)

    def _build_update(self, cluster_template):
        update = {}
        if self.params['is_floating_ip_enabled'] and self.params['external_network_id'] is None:
            raise ValueError('is_floating_ip_enabled is True but external_network_id is missing')
        non_updateable_keys = [k for k in ['coe', 'dns_nameserver', 'docker_storage_driver', 'docker_volume_size', 'external_network_id', 'fixed_network', 'fixed_subnet', 'flavor_id', 'http_proxy', 'https_proxy', 'image_id', 'is_floating_ip_enabled', 'is_master_lb_enabled', 'is_public', 'is_registry_enabled', 'is_tls_disabled', 'keypair_id', 'master_flavor_id', 'name', 'network_driver', 'no_proxy', 'server_type', 'volume_driver'] if self.params[k] is not None and self.params[k] != cluster_template[k]]
        labels = self.params['labels']
        if labels is not None:
            if isinstance(labels, str):
                labels = dict([tuple(kv.split(':')) for kv in labels.split(',')])
            if labels != cluster_template['labels']:
                non_updateable_keys.append('labels')
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in [] if self.params[k] is not None and self.params[k] != cluster_template[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        if self.params['is_floating_ip_enabled'] and self.params['external_network_id'] is None:
            raise ValueError('is_floating_ip_enabled is True but external_network_id is missing')
        kwargs = dict(((k, self.params[k]) for k in ['coe', 'dns_nameserver', 'docker_storage_driver', 'docker_volume_size', 'external_network_id', 'fixed_network', 'fixed_subnet', 'flavor_id', 'http_proxy', 'https_proxy', 'image_id', 'is_floating_ip_enabled', 'is_master_lb_enabled', 'is_public', 'is_registry_enabled', 'is_tls_disabled', 'keypair_id', 'master_flavor_id', 'name', 'network_driver', 'no_proxy', 'server_type', 'volume_driver'] if self.params[k] is not None))
        labels = self.params['labels']
        if labels is not None:
            if isinstance(labels, str):
                labels = dict([tuple(kv.split(':')) for kv in labels.split(',')])
            kwargs['labels'] = labels
        return self.conn.container_infrastructure_management.create_cluster_template(**kwargs)

    def _delete(self, cluster_template):
        self.conn.container_infrastructure_management.delete_cluster_template(cluster_template['id'])

    def _find(self):
        name = self.params['name']
        filters = {}
        image_id = self.params['image_id']
        if image_id is not None:
            filters['image_id'] = image_id
        coe = self.params['coe']
        if coe is not None:
            filters['coe'] = coe
        return self.conn.get_cluster_template(name_or_id=name, filters=filters)

    def _update(self, cluster_template, update):
        attributes = update.get('attributes')
        if attributes:
            pass
        return cluster_template

    def _will_change(self, state, cluster_template):
        if state == 'present' and (not cluster_template):
            return True
        elif state == 'present' and cluster_template:
            return bool(self._build_update(cluster_template))
        elif state == 'absent' and cluster_template:
            return True
        else:
            return False