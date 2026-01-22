import base64
import secrets
import uuid
import flask
import http.client
from oslo_serialization import jsonutils
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.application_credential import schema as app_cred_schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
import keystone.conf
from keystone import exception as ks_exception
from keystone.i18n import _
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class OAuth1AccessTokenRoleListResource(ks_flask.ResourceBase):
    collection_key = 'roles'
    member_key = 'role'

    def get(self, user_id, access_token_id):
        """List roles for a user access token.

        GET/HEAD /v3/users/{user_id}/OS-OAUTH1/access_tokens/
                 {access_token_id}/roles
        """
        ENFORCER.enforce_call(action='identity:list_access_token_roles')
        access_token = PROVIDERS.oauth_api.get_access_token(access_token_id)
        if access_token['authorizing_user_id'] != user_id:
            raise ks_exception.NotFound()
        authed_role_ids = access_token['role_ids']
        authed_role_ids = jsonutils.loads(authed_role_ids)
        refs = [_format_role_entity(x) for x in authed_role_ids]
        return self.wrap_collection(refs)