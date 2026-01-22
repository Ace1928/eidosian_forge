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
class ProjectUserListGrantResource(_ProjectGrantResourceBase):

    def get(self, project_id, user_id):
        """List grants for user on project.

        GET/HEAD /v3/projects/{project_id}/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:list_grants', build_target=functools.partial(self._build_enforcement_target_attr, project_id=project_id, user_id=user_id))
        inherited = self._check_if_inherited()
        refs = PROVIDERS.assignment_api.list_grants(user_id=user_id, project_id=project_id, inherited_to_projects=inherited)
        return self.wrap_collection(refs)