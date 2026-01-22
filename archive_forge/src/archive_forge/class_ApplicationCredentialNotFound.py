import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ApplicationCredentialNotFound(NotFound):
    message_format = _('Could not find Application Credential: %(application_credential_id)s.')