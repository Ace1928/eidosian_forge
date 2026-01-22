import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class LDAPSizeLimitExceeded(UnexpectedError):
    message_format = _('Number of User/Group entities returned by LDAP exceeded size limit. Contact your LDAP administrator.')