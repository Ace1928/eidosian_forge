import string
import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import strutils
import urllib
import werkzeug.exceptions
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.api._shared import saml
from keystone.auth import schema as auth_schema
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import utils as k_utils
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import schema as federation_schema
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
class AuthFederationWebSSOIDPsResource(_AuthFederationWebSSOBase):

    @classmethod
    def _perform_auth(cls, idp_id, protocol_id):
        host = _get_sso_origin_host()
        token = authentication.federated_authenticate_for_token(identity_provider=idp_id, protocol_id=protocol_id)
        return cls._render_template_response(host, token.id)

    @ks_flask.unenforced_api
    def get(self, idp_id, protocol_id):
        return self._perform_auth(idp_id, protocol_id)

    @ks_flask.unenforced_api
    def post(self, idp_id, protocol_id):
        return self._perform_auth(idp_id, protocol_id)