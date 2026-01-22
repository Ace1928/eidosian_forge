import base64
import datetime
import uuid
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.models import token_model
from keystone import notifications
def _validate_token_access_rules(self, token, access_rules_support=None):
    if token.application_credential_id:
        app_cred_api = PROVIDERS.application_credential_api
        app_cred = app_cred_api.get_application_credential(token.application_credential_id)
        if app_cred.get('access_rules') is not None and (not access_rules_support or float(access_rules_support) < ACCESS_RULES_MIN_VERSION):
            LOG.exception('Attempted to use application credential access rules with a middleware that does not understand them. You must upgrade keystonemiddleware on all services that accept application credentials as an authentication method.')
            raise exception.TokenNotFound(_('Failed to validate token'))