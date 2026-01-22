import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class MappedGroupNotFound(UnexpectedError):
    debug_message_format = _('Group %(group_id)s returned by mapping %(mapping_id)s was not found in the backend.')