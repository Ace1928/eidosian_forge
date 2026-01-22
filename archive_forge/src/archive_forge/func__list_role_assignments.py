import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _list_role_assignments(self):
    filters = ['group.id', 'role.id', 'scope.domain.id', 'scope.project.id', 'scope.OS-INHERIT:inherited_to', 'user.id', 'scope.system']
    target = None
    if self.oslo_context.domain_id:
        target = {'domain_id': self.oslo_context.domain_id}
    ENFORCER.enforce_call(action='identity:list_role_assignments', filters=filters, target_attr=target)
    assignments = self._build_role_assignments_list()
    if self.oslo_context.domain_id:
        domain_assignments = []
        for assignment in assignments['role_assignments']:
            domain_id = assignment['scope'].get('domain', {}).get('id')
            project_id = assignment['scope'].get('project', {}).get('id')
            if domain_id == self.oslo_context.domain_id:
                domain_assignments.append(assignment)
                continue
            elif project_id:
                project = PROVIDERS.resource_api.get_project(project_id)
                if project.get('domain_id') == self.oslo_context.domain_id:
                    domain_assignments.append(assignment)
        assignments['role_assignments'] = domain_assignments
    return assignments