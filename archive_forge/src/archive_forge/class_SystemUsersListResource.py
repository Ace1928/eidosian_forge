import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.server import flask as ks_flask
class SystemUsersListResource(flask_restful.Resource):

    def get(self, user_id):
        """List all system grants for a specific user.

        GET/HEAD /system/users/{user_id}/roles
        """
        ENFORCER.enforce_call(action='identity:list_system_grants_for_user', build_target=_build_enforcement_target)
        refs = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        return ks_flask.ResourceBase.wrap_collection(refs, collection_name='roles')