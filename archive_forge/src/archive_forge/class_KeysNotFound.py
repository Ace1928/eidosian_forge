import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class KeysNotFound(UnexpectedError):
    debug_message_format = _('No encryption keys found; run keystone-manage fernet_setup to bootstrap one.')