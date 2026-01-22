import flask_restful
from keystone.api._shared import json_home_relations
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class SimpleCertAPI(ks_flask.APIBase):
    _name = 'OS-SIMPLE-CERT'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=SimpleCertCAResource, url='/OS-SIMPLE-CERT/ca', resource_kwargs={}, rel='ca_certificate', resource_relation_func=_build_resource_relation), ks_flask.construct_resource_map(resource=SimpleCertListResource, url='/OS-SIMPLE-CERT/certificates', resource_kwargs={}, rel='certificates', resource_relation_func=_build_resource_relation)]