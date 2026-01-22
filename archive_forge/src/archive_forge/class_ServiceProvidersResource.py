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
class ServiceProvidersResource(_ResourceBase):
    collection_key = 'service_providers'
    member_key = 'service_provider'
    _public_parameters = frozenset(['auth_url', 'id', 'enabled', 'description', 'links', 'relay_state_prefix', 'sp_url'])
    _id_path_param_name_override = 'sp_id'
    api_prefix = '/OS-FEDERATION'

    def get(self, sp_id=None):
        if sp_id is not None:
            return self._get_service_provider(sp_id)
        return self._list_service_providers()

    def _get_service_provider(self, sp_id):
        """Get a service provider.

        GET/HEAD /OS-FEDERATION/service_providers/{sp_id}
        """
        ENFORCER.enforce_call(action='identity:get_service_provider')
        return self.wrap_member(PROVIDERS.federation_api.get_sp(sp_id))

    def _list_service_providers(self):
        """List service providers.

        GET/HEAD /OS-FEDERATION/service_providers
        """
        filters = ['id', 'enabled']
        ENFORCER.enforce_call(action='identity:list_service_providers', filters=filters)
        hints = self.build_driver_hints(filters)
        refs = [self.filter_params(r) for r in PROVIDERS.federation_api.list_sps(hints=hints)]
        return self.wrap_collection(refs, hints=hints)

    def put(self, sp_id):
        """Create a service provider.

        PUT /OS-FEDERATION/service_providers/{sp_id}
        """
        ENFORCER.enforce_call(action='identity:create_service_provider')
        sp = self.request_body_json.get('service_provider', {})
        validation.lazy_validate(schema.service_provider_create, sp)
        sp = self._normalize_dict(sp)
        sp.setdefault('enabled', False)
        sp.setdefault('relay_state_prefix', CONF.saml.relay_state_prefix)
        sp_ref = PROVIDERS.federation_api.create_sp(sp_id, sp)
        return (self.wrap_member(sp_ref), http.client.CREATED)

    def patch(self, sp_id):
        """Update a service provider.

        PATCH /OS-FEDERATION/service_providers/{sp_id}
        """
        ENFORCER.enforce_call(action='identity:update_service_provider')
        sp = self.request_body_json.get('service_provider', {})
        validation.lazy_validate(schema.service_provider_update, sp)
        sp = self._normalize_dict(sp)
        sp_ref = PROVIDERS.federation_api.update_sp(sp_id, sp)
        return self.wrap_member(sp_ref)

    def delete(self, sp_id):
        """Delete a service provider.

        DELETE /OS-FEDERATION/service_providers/{sp_id}
        """
        ENFORCER.enforce_call(action='identity:delete_service_provider')
        PROVIDERS.federation_api.delete_sp(sp_id)
        return (None, http.client.NO_CONTENT)