import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class MultipleSQLDriversInConfig(UnexpectedError):
    debug_message_format = _('The Keystone domain-specific configuration has specified more than one SQL driver (only one is permitted): %(source)s.')