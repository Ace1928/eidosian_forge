import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class InvalidOperatorError(ValidationError):
    message_format = _("The given operator %(_op)s is not valid. It must be one of the following: 'eq', 'neq', 'lt', 'lte', 'gt', or 'gte'.")