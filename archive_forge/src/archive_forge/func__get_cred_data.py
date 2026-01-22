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
def _get_cred_data(credential_id):
    cred = PROVIDERS.credential_api.get_credential(credential_id)
    if not cred or cred['type'] != CRED_TYPE_EC2:
        raise ks_exception.Unauthorized(message=_('EC2 access key not found.'))
    return _convert_v3_to_ec2_credential(cred)