import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class AssignmentTypeCalculationError(UnexpectedError):
    debug_message_format = _('Unexpected combination of grant attributes - User: %(user_id)s, Group: %(group_id)s, Project: %(project_id)s, Domain: %(domain_id)s.')