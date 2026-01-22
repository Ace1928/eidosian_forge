from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityRoleAssignmentModule(OpenStackModule):
    argument_spec = dict(domain=dict(), group=dict(), project=dict(), role=dict(required=True), state=dict(default='present', choices=['absent', 'present']), system=dict(), user=dict())
    module_kwargs = dict(required_one_of=[('user', 'group'), ('domain', 'project', 'system')], supports_check_mode=True)

    def run(self):
        filters = {}
        find_filters = {}
        kwargs = {}
        role_name_or_id = self.params['role']
        role = self.conn.identity.find_role(role_name_or_id, ignore_missing=False)
        filters['role_id'] = role['id']
        domain_name_or_id = self.params['domain']
        if domain_name_or_id is not None:
            domain = self.conn.identity.find_domain(domain_name_or_id, ignore_missing=False)
            filters['scope_domain_id'] = domain['id']
            find_filters['domain_id'] = domain['id']
            kwargs['domain'] = domain['id']
        user_name_or_id = self.params['user']
        if user_name_or_id is not None:
            user = self.conn.identity.find_user(user_name_or_id, ignore_missing=False, **find_filters)
            filters['user_id'] = user['id']
            kwargs['user'] = user['id']
        group_name_or_id = self.params['group']
        if group_name_or_id is not None:
            group = self.conn.identity.find_group(group_name_or_id, ignore_missing=False, **find_filters)
            filters['group_id'] = group['id']
            kwargs['group'] = group['id']
        system_name = self.params['system']
        if system_name is not None:
            if 'scope_domain_id' not in filters:
                filters['scope.system'] = system_name
            kwargs['system'] = system_name
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False, **find_filters)
            filters['scope_project_id'] = project['id']
            kwargs['project'] = project['id']
            filters.pop('scope_domain_id', None)
            filters.pop('scope.system', None)
        role_assignments = list(self.conn.identity.role_assignments(**filters))
        state = self.params['state']
        if self.ansible.check_mode:
            self.exit_json(changed=state == 'present' and (not role_assignments) or (state == 'absent' and role_assignments))
        if state == 'present' and (not role_assignments):
            self.conn.grant_role(role['id'], **kwargs)
            self.exit_json(changed=True)
        elif state == 'absent' and role_assignments:
            self.conn.revoke_role(role['id'], **kwargs)
            self.exit_json(changed=True)
        else:
            self.exit_json(changed=False)