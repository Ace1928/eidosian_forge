from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from collections import defaultdict
class QuotaModule(OpenStackModule):
    argument_spec = dict(backup_gigabytes=dict(type='int'), backups=dict(type='int'), cores=dict(type='int'), fixed_ips=dict(type='int'), floating_ips=dict(type='int', aliases=['floatingip', 'compute_floating_ips', 'network_floating_ips']), gigabytes=dict(type='int'), groups=dict(type='int'), injected_file_content_bytes=dict(type='int', aliases=['injected_file_size']), injected_file_path_bytes=dict(type='int', aliases=['injected_path_size']), injected_files=dict(type='int'), instances=dict(type='int'), key_pairs=dict(type='int', no_log=False), load_balancers=dict(type='int', aliases=['loadbalancer']), metadata_items=dict(type='int'), name=dict(required=True), networks=dict(type='int', aliases=['network']), per_volume_gigabytes=dict(type='int'), pools=dict(type='int', aliases=['pool']), ports=dict(type='int', aliases=['port']), ram=dict(type='int'), rbac_policies=dict(type='int', aliases=['rbac_policy']), routers=dict(type='int', aliases=['router']), security_group_rules=dict(type='int', aliases=['security_group_rule']), security_groups=dict(type='int', aliases=['security_group']), server_group_members=dict(type='int'), server_groups=dict(type='int'), snapshots=dict(type='int'), state=dict(default='present', choices=['absent', 'present']), subnet_pools=dict(type='int', aliases=['subnetpool']), subnets=dict(type='int', aliases=['subnet']), volumes=dict(type='int'))
    module_kwargs = dict(supports_check_mode=True)
    exclusion_map = {'compute': {'floating_ips', 'name', 'networks', 'security_group_rules', 'security_groups'}, 'network': {'name'}, 'volume': {'name'}}

    def _get_quotas(self, project):
        quota = {}
        if self.conn.has_service('block-storage'):
            quota['volume'] = self.conn.block_storage.get_quota_set(project)
        else:
            self.warn('Block storage service aka volume service is not supported by your cloud. Ignoring volume quotas.')
        if self.conn.has_service('network'):
            quota['network'] = self.conn.network.get_quota(project.id)
        else:
            self.warn('Network service is not supported by your cloud. Ignoring network quotas.')
        quota['compute'] = self.conn.compute.get_quota_set(project.id)
        return quota

    def _build_update(self, quotas):
        changes = defaultdict(dict)
        for quota_type in quotas.keys():
            exclusions = self.exclusion_map[quota_type]
            for attr in quotas[quota_type].keys():
                if attr in exclusions:
                    continue
                if attr in self.params and self.params[attr] is not None and (quotas[quota_type][attr] != self.params[attr]):
                    changes[quota_type][attr] = self.params[attr]
        return changes

    def _system_state_change(self, project_quota_output):
        """
        Determine if changes are required to the current project quota.

        This is done by comparing the current project_quota_output against
        the desired quota settings set on the module params.
        """
        if self.params['state'] == 'absent':
            return True
        return bool(self._build_update(project_quota_output))

    def run(self):
        project = self.conn.identity.find_project(self.params['name'], ignore_missing=False)
        quotas = self._get_quotas(project)
        changed = False
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(quotas))
        if self.params['state'] == 'absent':
            changed = True
            self.conn.compute.revert_quota_set(project)
            if 'network' in quotas:
                self.conn.network.delete_quota(project.id)
            if 'volume' in quotas:
                self.conn.block_storage.revert_quota_set(project)
            quotas = self._get_quotas(project)
        elif self.params['state'] == 'present':
            changes = self._build_update(quotas)
            if changes:
                if 'volume' in changes:
                    self.conn.block_storage.update_quota_set(quotas['volume'], **changes['volume'])
                if 'compute' in changes:
                    self.conn.compute.update_quota_set(quotas['compute'], **changes['compute'])
                if 'network' in changes:
                    quotas['network'] = self.conn.network.update_quota(project.id, **changes['network'])
                changed = True
        quotas = {k: v.to_dict(computed=False) for k, v in quotas.items()}
        self.exit_json(changed=changed, quotas=quotas)