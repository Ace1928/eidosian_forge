import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class IdentityProviderNotFound(NotFound):
    message_format = _('Could not find Identity Provider: %(idp_id)s.')