import string
import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import strutils
import urllib
import werkzeug.exceptions
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.api._shared import saml
from keystone.auth import schema as auth_schema
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import utils as k_utils
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import schema as federation_schema
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
class AuthFederationAPI(ks_flask.APIBase):
    _name = 'auth/OS-FEDERATION'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=AuthFederationSaml2Resource, url='/auth/OS-FEDERATION/saml2', resource_kwargs={}, resource_relation_func=json_home_relations.os_federation_resource_rel_func, rel='saml2'), ks_flask.construct_resource_map(resource=AuthFederationSaml2ECPResource, url='/auth/OS-FEDERATION/saml2/ecp', resource_kwargs={}, resource_relation_func=json_home_relations.os_federation_resource_rel_func, rel='ecp'), ks_flask.construct_resource_map(resource=AuthFederationWebSSOResource, url='/auth/OS-FEDERATION/websso/<string:protocol_id>', resource_kwargs={}, rel='websso', resource_relation_func=json_home_relations.os_federation_resource_rel_func, path_vars={'protocol_id': json_home_relations.os_federation_parameter_rel_func(parameter_name='protocol_id')}), ks_flask.construct_resource_map(resource=AuthFederationWebSSOIDPsResource, url='/auth/OS-FEDERATION/identity_providers/<string:idp_id>/protocols/<string:protocol_id>/websso', resource_kwargs={}, rel='identity_providers_websso', resource_relation_func=json_home_relations.os_federation_resource_rel_func, path_vars={'idp_id': json_home_relations.os_federation_parameter_rel_func(parameter_name='idp_id'), 'protocol_id': json_home_relations.os_federation_parameter_rel_func(parameter_name='protocol_id')})]