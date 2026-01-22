import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class CircularRegionHierarchyError(Error):
    message_format = _('The specified parent region %(parent_region_id)s would create a circular region hierarchy.')
    code = int(http.client.BAD_REQUEST)
    title = http.client.responses[http.client.BAD_REQUEST]