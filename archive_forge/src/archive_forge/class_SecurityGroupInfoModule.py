from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class SecurityGroupInfoModule(OpenStackModule):
    argument_spec = dict(any_tags=dict(type='list', elements='str'), description=dict(), name=dict(), not_any_tags=dict(type='list', elements='str'), not_tags=dict(type='list', elements='str'), project_id=dict(), revision_number=dict(type='int'), tags=dict(type='list', elements='str'))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        name = self.params['name']
        args = {k: self.params[k] for k in ['description', 'project_id', 'revision_number'] if self.params[k]}
        args.update({k: ','.join(self.params[k]) for k in ['tags', 'any_tags', 'not_tags', 'not_any_tags'] if self.params[k]})
        security_groups = self.conn.network.security_groups(**args)
        if name:
            security_groups = [item for item in security_groups if name in (item['id'], item['name'])]
        self.exit(changed=False, security_groups=[sg.to_dict(computed=False) for sg in security_groups])