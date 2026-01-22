import flask_restful
import http.client
from oslo_log import versionutils
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.policy import schema
from keystone.server import flask as ks_flask
class EndpointPolicyAssociations(flask_restful.Resource):

    def get(self, policy_id, endpoint_id):
        action = 'identity:check_policy_association_for_endpoint'
        ENFORCER.enforce_call(action=action)
        PROVIDERS.policy_api.get_policy(policy_id)
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        PROVIDERS.endpoint_policy_api.check_policy_association(policy_id, endpoint_id=endpoint_id)
        return (None, http.client.NO_CONTENT)

    def put(self, policy_id, endpoint_id):
        action = 'identity:create_policy_association_for_endpoint'
        ENFORCER.enforce_call(action=action)
        PROVIDERS.policy_api.get_policy(policy_id)
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        PROVIDERS.endpoint_policy_api.create_policy_association(policy_id, endpoint_id=endpoint_id)
        return (None, http.client.NO_CONTENT)

    def delete(self, policy_id, endpoint_id):
        action = 'identity:delete_policy_association_for_endpoint'
        ENFORCER.enforce_call(action=action)
        PROVIDERS.policy_api.get_policy(policy_id)
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        PROVIDERS.endpoint_policy_api.delete_policy_association(policy_id, endpoint_id=endpoint_id)
        return (None, http.client.NO_CONTENT)