import flask
from flask import make_response
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import utils
from keystone.conf import CONF
from keystone import exception
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
class OSAuth2API(ks_flask.APIBase):
    _name = 'OS-OAUTH2'
    _import_name = __name__
    _api_url_prefix = '/OS-OAUTH2'
    resource_mapping = [ks_flask.construct_resource_map(resource=AccessTokenResource, url='/token', rel='token', resource_kwargs={}, resource_relation_func=_build_resource_relation)]