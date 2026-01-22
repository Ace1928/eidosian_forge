import flask
import flask_restful
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.limit import schema
from keystone.server import flask as ks_flask
class LimitsAPI(ks_flask.APIBase):
    _name = 'limits'
    _import_name = __name__
    resources = [LimitsResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=LimitModelResource, resource_kwargs={}, url='/limits/model', rel='limit_model', status=json_home.Status.EXPERIMENTAL)]