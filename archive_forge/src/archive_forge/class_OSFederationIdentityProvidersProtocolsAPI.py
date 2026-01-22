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
class OSFederationIdentityProvidersProtocolsAPI(ks_flask.APIBase):
    _name = 'protocols'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=IDPProtocolsCRUDResource, url='/OS-FEDERATION/identity_providers/<string:idp_id>/protocols/<string:protocol_id>', resource_kwargs={}, rel='identity_provider_protocol', resource_relation_func=_build_resource_relation, path_vars={'idp_id': IDP_ID_PARAMETER_RELATION, 'protocol_id': PROTOCOL_ID_PARAMETER_RELATION}), ks_flask.construct_resource_map(resource=IDPProtocolsListResource, url='/OS-FEDERATION/identity_providers/<string:idp_id>/protocols', resource_kwargs={}, rel='identity_provider_protocols', resource_relation_func=_build_resource_relation, path_vars={'idp_id': IDP_ID_PARAMETER_RELATION})]