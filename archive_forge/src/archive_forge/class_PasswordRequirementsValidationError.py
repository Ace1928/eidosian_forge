import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class PasswordRequirementsValidationError(PasswordValidationError):
    message_format = _('The password does not match the requirements: %(detail)s.')