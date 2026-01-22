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
class IdentityProvidersResource(_ResourceBase):
    collection_key = 'identity_providers'
    member_key = 'identity_provider'
    api_prefix = '/OS-FEDERATION'
    _public_parameters = frozenset(['id', 'enabled', 'description', 'remote_ids', 'links', 'domain_id', 'authorization_ttl'])
    _id_path_param_name_override = 'idp_id'

    @staticmethod
    def _add_related_links(ref):
        """Add URLs for entities related with Identity Provider.

        Add URLs pointing to:
        - protocols tied to the Identity Provider

        """
        base_path = ref['links'].get('self')
        if base_path is None:
            base_path = '/'.join(ks_flask.base_url(path='/%s' % ref['id']))
        for name in ['protocols']:
            ref['links'][name] = '/'.join([base_path, name])

    def get(self, idp_id=None):
        if idp_id is not None:
            return self._get_idp(idp_id)
        return self._list_idps()

    def _get_idp(self, idp_id):
        """Get an IDP resource.

        GET/HEAD /OS-FEDERATION/identity_providers/{idp_id}
        """
        ENFORCER.enforce_call(action='identity:get_identity_provider')
        ref = PROVIDERS.federation_api.get_idp(idp_id)
        return self.wrap_member(ref)

    def _list_idps(self):
        """List all identity providers.

        GET/HEAD /OS-FEDERATION/identity_providers
        """
        filters = ['id', 'enabled']
        ENFORCER.enforce_call(action='identity:list_identity_providers', filters=filters)
        hints = self.build_driver_hints(filters)
        refs = PROVIDERS.federation_api.list_idps(hints=hints)
        refs = [self.filter_params(r) for r in refs]
        collection = self.wrap_collection(refs, hints=hints)
        for r in collection[self.collection_key]:
            self._add_related_links(r)
        return collection

    def put(self, idp_id):
        """Create an idp resource for federated authentication.

        PUT /OS-FEDERATION/identity_providers/{idp_id}
        """
        ENFORCER.enforce_call(action='identity:create_identity_provider')
        idp = self.request_body_json.get('identity_provider', {})
        validation.lazy_validate(schema.identity_provider_create, idp)
        idp = self._normalize_dict(idp)
        idp.setdefault('enabled', False)
        idp_ref = PROVIDERS.federation_api.create_idp(idp_id, idp)
        return (self.wrap_member(idp_ref), http.client.CREATED)

    def patch(self, idp_id):
        ENFORCER.enforce_call(action='identity:update_identity_provider')
        idp = self.request_body_json.get('identity_provider', {})
        validation.lazy_validate(schema.identity_provider_update, idp)
        idp = self._normalize_dict(idp)
        idp_ref = PROVIDERS.federation_api.update_idp(idp_id, idp)
        return self.wrap_member(idp_ref)

    def delete(self, idp_id):
        ENFORCER.enforce_call(action='identity:delete_identity_provider')
        PROVIDERS.federation_api.delete_idp(idp_id)
        return (None, http.client.NO_CONTENT)