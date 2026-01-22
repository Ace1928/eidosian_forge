from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _build_grant_filter(self, session, role_id, user_id, group_id, domain_id, project_id, inherited_to_projects):
    q = session.query(RoleAssignment)
    q = q.filter_by(actor_id=user_id or group_id)
    if domain_id:
        q = q.filter_by(target_id=domain_id).filter((RoleAssignment.type == AssignmentType.USER_DOMAIN) | (RoleAssignment.type == AssignmentType.GROUP_DOMAIN))
    else:
        q = q.filter_by(target_id=project_id).filter((RoleAssignment.type == AssignmentType.USER_PROJECT) | (RoleAssignment.type == AssignmentType.GROUP_PROJECT))
    q = q.filter_by(role_id=role_id)
    q = q.filter_by(inherited=inherited_to_projects)
    return q