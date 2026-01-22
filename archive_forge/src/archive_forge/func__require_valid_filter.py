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
@staticmethod
def _require_valid_filter(endpoint_group):
    valid_filter_keys = ['service_id', 'region_id', 'interface']
    filters = endpoint_group.get('filters')
    for key in filters.keys():
        if key not in valid_filter_keys:
            raise exception.ValidationError(attribute=' or '.join(valid_filter_keys), target='endpoint_group')