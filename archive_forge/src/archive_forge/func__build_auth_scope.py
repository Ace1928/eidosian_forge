from keystone.i18n import _
def _build_auth_scope(self, system=False, project_id=None, project_name=None, project_domain_id=None, project_domain_name=None, domain_id=None, domain_name=None, trust_id=None, unscoped=None):
    scope_data = {}
    if system:
        scope_data['system'] = {'all': True}
    elif unscoped:
        scope_data['unscoped'] = {}
    elif project_id or project_name:
        scope_data['project'] = {}
        if project_id:
            scope_data['project']['id'] = project_id
        else:
            scope_data['project']['name'] = project_name
            if project_domain_id or project_domain_name:
                project_domain_json = {}
                if project_domain_id:
                    project_domain_json['id'] = project_domain_id
                else:
                    project_domain_json['name'] = project_domain_name
                scope_data['project']['domain'] = project_domain_json
    elif domain_id or domain_name:
        scope_data['domain'] = {}
        if domain_id:
            scope_data['domain']['id'] = domain_id
        else:
            scope_data['domain']['name'] = domain_name
    elif trust_id:
        scope_data['OS-TRUST:trust'] = {}
        scope_data['OS-TRUST:trust']['id'] = trust_id
    else:
        raise ValueError(_('Programming Error: Invalid arguments supplied to build scope.'))
    return scope_data