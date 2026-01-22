import hashlib
import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.credential import schema
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class CredentialAPI(ks_flask.APIBase):
    _name = 'credentials'
    _import_name = __name__
    resource_mapping = []
    resources = [CredentialResource]