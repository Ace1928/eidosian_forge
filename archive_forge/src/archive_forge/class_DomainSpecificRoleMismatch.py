import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class DomainSpecificRoleMismatch(Forbidden):
    message_format = _('Project %(project_id)s must be in the same domain as the role %(role_id)s being assigned.')