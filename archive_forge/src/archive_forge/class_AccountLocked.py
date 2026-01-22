import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class AccountLocked(Unauthorized):
    message_format = _('The account is locked for user: %(user_id)s.')