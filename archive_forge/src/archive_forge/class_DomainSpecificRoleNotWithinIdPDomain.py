import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class DomainSpecificRoleNotWithinIdPDomain(Forbidden):
    message_format = _('role: %(role_name)s must be within the same domain as the identity provider: %(identity_provider)s.')