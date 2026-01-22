import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class RoleAssignmentNotFound(NotFound):
    message_format = _('Could not find role assignment with role: %(role_id)s, user or group: %(actor_id)s, project, domain, or system: %(target_id)s.')