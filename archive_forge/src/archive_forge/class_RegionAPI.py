import http.client
from keystone.catalog import schema
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class RegionAPI(ks_flask.APIBase):
    _name = 'regions'
    _import_name = __name__
    resources = [RegionResource]
    resource_mapping = []