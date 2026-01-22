import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class NoLimitReference(Forbidden):
    message_format = _('Unable to create a limit that has no corresponding registered limit.')