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
class AuthTokenOSPKIResource(flask_restful.Resource):

    @ks_flask.unenforced_api
    def get(self):
        """Deprecated; get revoked token list.

        GET/HEAD /v3/auth/tokens/OS-PKI/revoked
        """
        if not CONF.token.revoke_by_id:
            raise exception.Gone()
        raise exception.Forbidden()