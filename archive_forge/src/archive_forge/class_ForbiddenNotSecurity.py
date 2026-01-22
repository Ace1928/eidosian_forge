import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ForbiddenNotSecurity(Error):
    """When you want to return a 403 Forbidden response but not security.

    Use this for errors where the message is always safe to present to the user
    and won't give away extra information.

    """
    code = int(http.client.FORBIDDEN)
    title = http.client.responses[http.client.FORBIDDEN]