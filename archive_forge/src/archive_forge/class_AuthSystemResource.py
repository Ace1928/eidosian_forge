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
class AuthSystemResource(_AuthFederationWebSSOBase):

    def get(self):
        """Get possible system scopes for token.

        GET/HEAD /v3/auth/system
        """
        ENFORCER.enforce_call(action='identity:get_auth_system')
        user_id = self.auth_context.get('user_id')
        group_ids = self.auth_context.get('group_ids')
        user_assignments = []
        group_assignments = []
        if user_id:
            try:
                user_assignments = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
            except exception.UserNotFound:
                pass
        if group_ids:
            group_assignments = PROVIDERS.assignment_api.list_system_grants_for_groups(group_ids)
        assignments = _combine_lists_uniquely(user_assignments, group_assignments)
        if assignments:
            response = {'system': [{'all': True}], 'links': {'self': ks_flask.base_url(path='auth/system')}}
        else:
            response = {'system': [], 'links': {'self': ks_flask.base_url(path='auth/system')}}
        return response