import flask
import flask_restful
import http.client
from keystone.api._shared import implied_roles as shared
from keystone.assignment import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.server import flask as ks_flask
class RoleImplicationResource(flask_restful.Resource):

    def head(self, prior_role_id, implied_role_id=None):
        ENFORCER.enforce_call(action='identity:check_implied_role', build_target=_build_enforcement_target_ref)
        self.get(prior_role_id, implied_role_id)
        return (None, http.client.NO_CONTENT)

    def get(self, prior_role_id, implied_role_id):
        """Get implied role.

        GET/HEAD /v3/roles/{prior_role_id}/implies/{implied_role_id}
        """
        ENFORCER.enforce_call(action='identity:get_implied_role', build_target=_build_enforcement_target_ref)
        return self._get_implied_role(prior_role_id, implied_role_id)

    def _get_implied_role(self, prior_role_id, implied_role_id):
        PROVIDERS.role_api.get_implied_role(prior_role_id, implied_role_id)
        implied_role_ref = PROVIDERS.role_api.get_role(implied_role_id)
        response_json = shared.role_inference_response(prior_role_id)
        response_json['role_inference']['implies'] = shared.build_implied_role_response_data(implied_role_ref)
        response_json['links'] = {'self': ks_flask.base_url(path='/roles/%(prior)s/implies/%(implies)s' % {'prior': prior_role_id, 'implies': implied_role_id})}
        return response_json

    def put(self, prior_role_id, implied_role_id):
        """Create implied role.

        PUT /v3/roles/{prior_role_id}/implies/{implied_role_id}
        """
        ENFORCER.enforce_call(action='identity:create_implied_role', build_target=_build_enforcement_target_ref)
        PROVIDERS.role_api.create_implied_role(prior_role_id, implied_role_id)
        response_json = self._get_implied_role(prior_role_id, implied_role_id)
        return (response_json, http.client.CREATED)

    def delete(self, prior_role_id, implied_role_id):
        """Delete implied role.

        DELETE /v3/roles/{prior_role_id}/implies/{implied_role_id}
        """
        ENFORCER.enforce_call(action='identity:delete_implied_role', build_target=_build_enforcement_target_ref)
        PROVIDERS.role_api.delete_implied_role(prior_role_id, implied_role_id)
        return (None, http.client.NO_CONTENT)