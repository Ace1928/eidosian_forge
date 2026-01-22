import flask
import flask_restful
from oslo_utils import timeutils
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class OSRevokeAPI(ks_flask.APIBase):
    _name = 'events'
    _import_name = __name__
    _api_url_prefix = '/OS-REVOKE'
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=OSRevokeResource, url='/events', resource_kwargs={}, rel='events', resource_relation_func=_build_resource_relation)]