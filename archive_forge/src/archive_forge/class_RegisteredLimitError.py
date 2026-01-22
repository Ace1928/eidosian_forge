import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class RegisteredLimitError(ForbiddenNotSecurity):
    message_format = _('Unable to update or delete registered limit %(id)s because there are project limits associated with it.')