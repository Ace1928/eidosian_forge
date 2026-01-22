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
class UserResource(ks_flask.ResourceBase):
    collection_key = 'users'
    member_key = 'user'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='identity_api', method='get_user')

    def get(self, user_id=None):
        """Get a user resource or list users.

        GET/HEAD /v3/users
        GET/HEAD /v3/users/{user_id}
        """
        if user_id is not None:
            return self._get_user(user_id)
        return self._list_users()

    def _get_user(self, user_id):
        """Get a user resource.

        GET/HEAD /v3/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:get_user', build_target=_build_user_target_enforcement)
        ref = PROVIDERS.identity_api.get_user(user_id)
        return self.wrap_member(ref)

    def _list_users(self):
        """List users.

        GET/HEAD /v3/users
        """
        filters = ('domain_id', 'enabled', 'idp_id', 'name', 'protocol_id', 'unique_id', 'password_expires_at')
        target = None
        if self.oslo_context.domain_id:
            target = {'domain_id': self.oslo_context.domain_id}
        hints = self.build_driver_hints(filters)
        ENFORCER.enforce_call(action='identity:list_users', filters=filters, target_attr=target)
        domain = self._get_domain_id_for_list_request()
        if domain is None and self.oslo_context.domain_id:
            domain = self.oslo_context.domain_id
        refs = PROVIDERS.identity_api.list_users(domain_scope=domain, hints=hints)
        if self.oslo_context.domain_id:
            domain_id = self.oslo_context.domain_id
            users = [user for user in refs if user['domain_id'] == domain_id]
        else:
            users = refs
        return self.wrap_collection(users, hints=hints)

    def post(self):
        """Create a user.

        POST /v3/users
        """
        user_data = self.request_body_json.get('user', {})
        target = {'user': user_data}
        ENFORCER.enforce_call(action='identity:create_user', target_attr=target)
        validation.lazy_validate(schema.user_create, user_data)
        user_data = self._normalize_dict(user_data)
        user_data = self._normalize_domain_id(user_data)
        ref = PROVIDERS.identity_api.create_user(user_data, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    def patch(self, user_id):
        """Update a user.

        PATCH /v3/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:update_user', build_target=_build_user_target_enforcement)
        PROVIDERS.identity_api.get_user(user_id)
        user_data = self.request_body_json.get('user', {})
        validation.lazy_validate(schema.user_update, user_data)
        self._require_matching_id(user_data)
        ref = PROVIDERS.identity_api.update_user(user_id, user_data, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    def delete(self, user_id):
        """Delete a user.

        DELETE /v3/users/{user_id}
        """
        ENFORCER.enforce_call(action='identity:delete_user', build_target=_build_user_target_enforcement)
        PROVIDERS.identity_api.delete_user(user_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)