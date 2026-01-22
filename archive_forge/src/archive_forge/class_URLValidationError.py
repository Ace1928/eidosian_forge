import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class URLValidationError(ValidationError):
    message_format = _('Cannot create an endpoint with an invalid URL: %(url)s.')