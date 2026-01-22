import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class PasswordExpired(Unauthorized):
    message_format = _('The password is expired and needs to be changed for user: %(user_id)s.')