import functools
import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.resource import schema
from keystone.server import flask as ks_flask
class ProjectGroupGrantResource(_ProjectGrantResourceBase):

    def get(self, project_id, group_id, role_id):
        """Check grant for project, group, role.

        GET/HEAD /v3/projects/{project_id/groups/{group_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:check_grant', build_target=functools.partial(self._build_enforcement_target_attr, role_id=role_id, project_id=project_id, group_id=group_id))
        inherited = self._check_if_inherited()
        PROVIDERS.assignment_api.get_grant(role_id=role_id, group_id=group_id, project_id=project_id, inherited_to_projects=inherited)
        return (None, http.client.NO_CONTENT)

    def put(self, project_id, group_id, role_id):
        """Grant role for group on project.

        PUT /v3/projects/{project_id}/groups/{group_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:create_grant', build_target=functools.partial(self._build_enforcement_target_attr, role_id=role_id, project_id=project_id, group_id=group_id))
        inherited = self._check_if_inherited()
        PROVIDERS.assignment_api.create_grant(role_id=role_id, group_id=group_id, project_id=project_id, inherited_to_projects=inherited, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)

    def delete(self, project_id, group_id, role_id):
        """Delete grant of role for group on project.

        DELETE /v3/projects/{project_id}/groups/{group_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:revoke_grant', build_target=functools.partial(self._build_enforcement_target_attr, role_id=role_id, group_id=group_id, project_id=project_id, allow_non_existing=True))
        inherited = self._check_if_inherited()
        PROVIDERS.assignment_api.delete_grant(role_id=role_id, group_id=group_id, project_id=project_id, inherited_to_projects=inherited, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)