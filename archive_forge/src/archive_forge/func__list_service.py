import http.client
from keystone.catalog import schema
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.server import flask as ks_flask
def _list_service(self):
    filters = ['type', 'name']
    ENFORCER.enforce_call(action='identity:list_services', filters=filters)
    hints = self.build_driver_hints(filters)
    refs = PROVIDERS.catalog_api.list_services(hints=hints)
    return self.wrap_collection(refs, hints=hints)