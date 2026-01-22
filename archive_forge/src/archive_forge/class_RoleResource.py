import flask
import flask_restful
import http.client
from keystone.api._shared import implied_roles as shared
from keystone.assignment import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.server import flask as ks_flask
class RoleResource(ks_flask.ResourceBase):
    collection_key = 'roles'
    member_key = 'role'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='role_api', method='get_role')

    def _is_domain_role(self, role):
        return bool(role.get('domain_id'))

    def get(self, role_id=None):
        """Get role or list roles.

        GET/HEAD /v3/roles
        GET/HEAD /v3/roles/{role_id}
        """
        if role_id is not None:
            return self._get_role(role_id)
        return self._list_roles()

    def _get_role(self, role_id):
        err = None
        role = {}
        try:
            role = PROVIDERS.role_api.get_role(role_id)
        except Exception as e:
            err = e
        finally:
            if err is not None or not self._is_domain_role(role):
                ENFORCER.enforce_call(action='identity:get_role')
                if err:
                    raise err
            else:
                ENFORCER.enforce_call(action='identity:get_domain_role', member_target_type='role', member_target=role)
        return self.wrap_member(role)

    def _list_roles(self):
        filters = ['name', 'domain_id']
        domain_filter = flask.request.args.get('domain_id')
        if domain_filter:
            ENFORCER.enforce_call(action='identity:list_domain_roles', filters=filters)
        else:
            ENFORCER.enforce_call(action='identity:list_roles', filters=filters)
        hints = self.build_driver_hints(filters)
        if not domain_filter:
            hints.add_filter('domain_id', None)
        refs = PROVIDERS.role_api.list_roles(hints=hints)
        return self.wrap_collection(refs, hints=hints)

    def post(self):
        """Create role.

        POST /v3/roles
        """
        role = self.request_body_json.get('role', {})
        if self._is_domain_role(role):
            ENFORCER.enforce_call(action='identity:create_domain_role')
        else:
            ENFORCER.enforce_call(action='identity:create_role')
        validation.lazy_validate(schema.role_create, role)
        role = self._assign_unique_id(role)
        role = self._normalize_dict(role)
        ref = PROVIDERS.role_api.create_role(role['id'], role, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    def patch(self, role_id):
        """Update role.

        PATCH /v3/roles/{role_id}
        """
        err = None
        role = {}
        try:
            role = PROVIDERS.role_api.get_role(role_id)
        except Exception as e:
            err = e
        finally:
            if err is not None or not self._is_domain_role(role):
                ENFORCER.enforce_call(action='identity:update_role')
                if err:
                    raise err
            else:
                ENFORCER.enforce_call(action='identity:update_domain_role', member_target_type='role', member_target=role)
        request_body_role = self.request_body_json.get('role', {})
        validation.lazy_validate(schema.role_update, request_body_role)
        self._require_matching_id(request_body_role)
        ref = PROVIDERS.role_api.update_role(role_id, request_body_role, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    def delete(self, role_id):
        """Delete role.

        DELETE /v3/roles/{role_id}
        """
        err = None
        role = {}
        try:
            role = PROVIDERS.role_api.get_role(role_id)
        except Exception as e:
            err = e
        finally:
            if err is not None or not self._is_domain_role(role):
                ENFORCER.enforce_call(action='identity:delete_role')
                if err:
                    raise err
            else:
                ENFORCER.enforce_call(action='identity:delete_domain_role', member_target_type='role', member_target=role)
        PROVIDERS.role_api.delete_role(role_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)