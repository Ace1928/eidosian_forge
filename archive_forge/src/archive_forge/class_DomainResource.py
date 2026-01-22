import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.resource import schema
from keystone.server import flask as ks_flask
class DomainResource(ks_flask.ResourceBase):
    collection_key = 'domains'
    member_key = 'domain'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='resource_api', method='get_domain')

    def get(self, domain_id=None):
        """Get domain or list domains.

        GET/HEAD /v3/domains
        GET/HEAD /v3/domains/{domain_id}
        """
        if domain_id is not None:
            return self._get_domain(domain_id)
        return self._list_domains()

    def _get_domain(self, domain_id):
        ENFORCER.enforce_call(action='identity:get_domain', build_target=_build_domain_enforcement_target)
        domain = PROVIDERS.resource_api.get_domain(domain_id)
        return self.wrap_member(domain)

    def _list_domains(self):
        filters = ['name', 'enabled']
        target = None
        if self.oslo_context.domain_id:
            target = {'domain': {'id': self.oslo_context.domain_id}}
        ENFORCER.enforce_call(action='identity:list_domains', filters=filters, target_attr=target)
        hints = self.build_driver_hints(filters)
        refs = PROVIDERS.resource_api.list_domains(hints=hints)
        if self.oslo_context.domain_id:
            domain_id = self.oslo_context.domain_id
            filtered_refs = [ref for ref in refs if ref['id'] == domain_id]
        else:
            filtered_refs = refs
        return self.wrap_collection(filtered_refs, hints=hints)

    def post(self):
        """Create domain.

        POST /v3/domains
        """
        ENFORCER.enforce_call(action='identity:create_domain')
        domain = self.request_body_json.get('domain', {})
        validation.lazy_validate(schema.domain_create, domain)
        domain_id = domain.get('explicit_domain_id')
        if domain_id is None:
            domain = self._assign_unique_id(domain)
        else:
            try:
                self._validate_id_format(domain_id)
            except ValueError:
                raise exception.DomainIdInvalid
            domain['id'] = domain_id
        domain = self._normalize_dict(domain)
        ref = PROVIDERS.resource_api.create_domain(domain['id'], domain, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    def patch(self, domain_id):
        """Update domain.

        PATCH /v3/domains/{domain_id}
        """
        ENFORCER.enforce_call(action='identity:update_domain')
        domain = self.request_body_json.get('domain', {})
        validation.lazy_validate(schema.domain_update, domain)
        PROVIDERS.resource_api.get_domain(domain_id)
        ref = PROVIDERS.resource_api.update_domain(domain_id, domain, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    def delete(self, domain_id):
        """Delete domain.

        DELETE /v3/domains/{domain_id}
        """
        ENFORCER.enforce_call(action='identity:delete_domain')
        PROVIDERS.resource_api.delete_domain(domain_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)