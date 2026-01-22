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
class OSFederationMappingsAPI(ks_flask.APIBase):
    _name = 'mappings'
    _import_name = __name__
    _api_url_prefix = '/OS-FEDERATION'
    resources = [MappingResource]
    resource_mapping = []