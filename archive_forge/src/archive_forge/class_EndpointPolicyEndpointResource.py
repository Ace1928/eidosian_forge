import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
from keystone import exception
from keystone import notifications
from keystone.server import flask as ks_flask
class EndpointPolicyEndpointResource(flask_restful.Resource):

    def get(self, endpoint_id):
        ENFORCER.enforce_call(action='identity:get_policy_for_endpoint')
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        ref = PROVIDERS.endpoint_policy_api.get_policy_for_endpoint(endpoint_id)
        return ks_flask.ResourceBase.wrap_member(ref, collection_name='endpoints', member_name='policy')