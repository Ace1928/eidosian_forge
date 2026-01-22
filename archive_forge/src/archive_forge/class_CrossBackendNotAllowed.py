import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class CrossBackendNotAllowed(Forbidden):
    message_format = _('Group membership across backend boundaries is not allowed. Group in question is %(group_id)s, user is %(user_id)s.')