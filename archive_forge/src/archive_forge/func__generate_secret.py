import base64
import secrets
import uuid
import flask
import http.client
from oslo_serialization import jsonutils
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.application_credential import schema as app_cred_schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
import keystone.conf
from keystone import exception as ks_exception
from keystone.i18n import _
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
@staticmethod
def _generate_secret():
    length = 64
    secret = secrets.token_bytes(length)
    secret = base64.urlsafe_b64encode(secret)
    secret = secret.rstrip(b'=')
    secret = secret.decode('utf-8')
    return secret