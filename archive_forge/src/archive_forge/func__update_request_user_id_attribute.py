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
def _update_request_user_id_attribute():
    try:
        app_cred = PROVIDERS.application_credential_api.get_application_credential(flask.request.view_args.get('application_credential_id'))
        flask.request.view_args['user_id'] = app_cred['user_id']
        return {'user_id': app_cred['user_id']}
    except ks_exception.NotFound:
        pass