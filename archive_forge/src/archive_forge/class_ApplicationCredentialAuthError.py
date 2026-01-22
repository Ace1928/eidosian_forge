import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ApplicationCredentialAuthError(AuthPluginException):
    message_format = _('Error authenticating with application credential: %(detail)s')