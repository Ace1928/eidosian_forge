import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
from keystone import exception
from keystone import notifications
from keystone.server import flask as ks_flask
def _filter_endpoint(ref):
    ref.pop('legacy_endpoint_id', None)
    ref['region'] = ref['region_id']
    return ref