import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.api import endpoints as _endpoints_api
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _list_endpoint_groups(self):
    filters = 'name'
    ENFORCER.enforce_call(action='identity:list_endpoint_groups', filters=filters)
    hints = self.build_driver_hints(filters)
    refs = PROVIDERS.catalog_api.list_endpoint_groups(hints)
    return self.wrap_collection(refs, hints=hints)