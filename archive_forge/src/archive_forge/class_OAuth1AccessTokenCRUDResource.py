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
class OAuth1AccessTokenCRUDResource(_OAuth1ResourceBase):

    def get(self, user_id, access_token_id):
        """Get specific access token.

        GET/HEAD /v3/users/{user_id}/OS-OAUTH1/access_tokens/{access_token_id}
        """
        ENFORCER.enforce_call(action='identity:get_access_token')
        access_token = PROVIDERS.oauth_api.get_access_token(access_token_id)
        if access_token['authorizing_user_id'] != user_id:
            raise ks_exception.NotFound()
        access_token = _format_token_entity(access_token)
        return self.wrap_member(access_token)

    def delete(self, user_id, access_token_id):
        """Delete specific access token.

        DELETE /v3/users/{user_id}/OS-OAUTH1/access_tokens/{access_token_id}
        """
        ENFORCER.enforce_call(action='identity:ec2_delete_credential', build_target=_build_enforcer_target_data_owner_and_user_id_match)
        access_token = PROVIDERS.oauth_api.get_access_token(access_token_id)
        reason = 'Invalidating the token cache because an access token for consumer %(consumer_id)s has been deleted. Authorization for users with OAuth tokens will be recalculated and enforced accordingly the next time they authenticate or validate a token.' % {'consumer_id': access_token['consumer_id']}
        notifications.invalidate_token_cache_notification(reason)
        PROVIDERS.oauth_api.delete_access_token(user_id, access_token_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)