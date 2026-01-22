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
def _get_roles(self, app_cred_data, token):
    if app_cred_data.get('roles'):
        roles = self._normalize_role_list(app_cred_data['roles'])
        token_roles = [r['id'] for r in token.roles]
        for role in roles:
            if role['id'] not in token_roles:
                detail = _('Cannot create an application credential with unassigned role')
                raise ks_exception.ApplicationCredentialValidationError(detail=detail)
    else:
        roles = token.roles
    return roles