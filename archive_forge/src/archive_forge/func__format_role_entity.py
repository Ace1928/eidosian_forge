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
def _format_role_entity(role_id):
    role = PROVIDERS.role_api.get_role(role_id)
    formatted_entity = role.copy()
    if 'description' in role:
        formatted_entity.pop('description')
    if 'enabled' in role:
        formatted_entity.pop('enabled')
    return formatted_entity