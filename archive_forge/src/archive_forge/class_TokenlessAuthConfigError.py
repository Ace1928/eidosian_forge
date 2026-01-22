import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class TokenlessAuthConfigError(ValidationError):
    message_format = _('Could not determine Identity Provider ID. The configuration option %(issuer_attribute)s was not found in the request environment.')