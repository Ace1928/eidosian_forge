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
class _AuthFederationWebSSOBase(ks_flask.ResourceBase):

    @staticmethod
    def _render_template_response(host, token_id):
        with open(CONF.federation.sso_callback_template) as template:
            src = string.Template(template.read())
        subs = {'host': host, 'token': token_id}
        body = src.substitute(subs)
        resp = flask.make_response(body, http.client.OK)
        resp.charset = 'utf-8'
        resp.headers['Content-Type'] = 'text/html'
        return resp