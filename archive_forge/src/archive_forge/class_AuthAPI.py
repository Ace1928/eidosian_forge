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
class AuthAPI(ks_flask.APIBase):
    _name = 'auth'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=AuthProjectsResource, url='/auth/projects', alternate_urls=[dict(url='/OS-FEDERATION/projects', json_home=ks_flask.construct_json_home_data(rel='projects', resource_relation_func=json_home_relations.os_federation_resource_rel_func))], rel='auth_projects', resource_kwargs={}), ks_flask.construct_resource_map(resource=AuthDomainsResource, url='/auth/domains', alternate_urls=[dict(url='/OS-FEDERATION/domains', json_home=ks_flask.construct_json_home_data(rel='domains', resource_relation_func=json_home_relations.os_federation_resource_rel_func))], rel='auth_domains', resource_kwargs={}), ks_flask.construct_resource_map(resource=AuthSystemResource, url='/auth/system', resource_kwargs={}, rel='auth_system'), ks_flask.construct_resource_map(resource=AuthCatalogResource, url='/auth/catalog', resource_kwargs={}, rel='auth_catalog'), ks_flask.construct_resource_map(resource=AuthTokenOSPKIResource, url='/auth/tokens/OS-PKI/revoked', resource_kwargs={}, rel='revocations', resource_relation_func=json_home_relations.os_pki_resource_rel_func), ks_flask.construct_resource_map(resource=AuthTokenResource, url='/auth/tokens', resource_kwargs={}, rel='auth_tokens')]