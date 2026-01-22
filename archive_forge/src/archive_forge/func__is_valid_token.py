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
def _is_valid_token(self, token, window_seconds=0):
    """Verify the token is valid format and has not expired."""
    current_time = timeutils.normalize_time(timeutils.utcnow())
    try:
        expiry = timeutils.parse_isotime(token.expires_at)
        expiry = timeutils.normalize_time(expiry)
        expiry += datetime.timedelta(seconds=window_seconds)
    except Exception:
        LOG.exception('Unexpected error or malformed token determining token expiry: %s', token)
        raise exception.TokenNotFound(_('Failed to validate token'))
    if current_time < expiry:
        self.check_revocation(token)
        return None
    else:
        raise exception.TokenNotFound(_('Failed to validate token'))