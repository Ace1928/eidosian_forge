from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _normalize_to_id(self, role_assignment_prps):
    role_assignments = []
    if role_assignment_prps is None:
        return role_assignments
    for role_assignment in role_assignment_prps:
        role = role_assignment.get(self.ROLE)
        project = role_assignment.get(self.PROJECT)
        domain = role_assignment.get(self.DOMAIN)
        role_assignments.append({self.ROLE: self.client_plugin().get_role_id(role), self.PROJECT: self.client_plugin().get_project_id(project) if project else None, self.DOMAIN: self.client_plugin().get_domain_id(domain) if domain else None})
    return role_assignments