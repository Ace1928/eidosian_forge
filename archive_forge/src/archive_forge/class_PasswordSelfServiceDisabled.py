import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class PasswordSelfServiceDisabled(PasswordValidationError):
    message_format = _('You cannot change your password at this time due to password policy disallowing password changes. Please contact your administrator to reset your password.')