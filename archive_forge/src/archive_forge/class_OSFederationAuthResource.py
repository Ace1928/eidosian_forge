import flask
import flask_restful
import http.client
from oslo_serialization import jsonutils
from oslo_log import log
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import schema
from keystone.federation import utils
from keystone.server import flask as ks_flask
class OSFederationAuthResource(flask_restful.Resource):

    @ks_flask.unenforced_api
    def get(self, idp_id, protocol_id):
        """Authenticate from dedicated uri endpoint.

        GET/HEAD /OS-FEDERATION/identity_providers/
                 {idp_id}/protocols/{protocol_id}/auth
        """
        return self._auth(idp_id, protocol_id)

    @ks_flask.unenforced_api
    def post(self, idp_id, protocol_id):
        """Authenticate from dedicated uri endpoint.

        POST /OS-FEDERATION/identity_providers/
             {idp_id}/protocols/{protocol_id}/auth
        """
        return self._auth(idp_id, protocol_id)

    def _auth(self, idp_id, protocol_id):
        """Build and pass auth data to authentication code.

        Build HTTP request body for federated authentication and inject
        it into the ``authenticate_for_token`` function.
        """
        auth = {'identity': {'methods': [protocol_id], protocol_id: {'identity_provider': idp_id, 'protocol': protocol_id}}}
        token = authentication.authenticate_for_token(auth)
        token_data = render_token.render_token_response_from_model(token)
        resp_data = jsonutils.dumps(token_data)
        flask_resp = flask.make_response(resp_data, http.client.CREATED)
        flask_resp.headers['X-Subject-Token'] = token.id
        flask_resp.headers['Content-Type'] = 'application/json'
        return flask_resp