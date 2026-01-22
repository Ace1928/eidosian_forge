import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class InvalidImpliedRole(Forbidden):
    message_format = _('%(role_id)s cannot be an implied roles.')