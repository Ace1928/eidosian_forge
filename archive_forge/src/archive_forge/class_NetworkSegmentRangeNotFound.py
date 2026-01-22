from neutron_lib._i18n import _
from neutron_lib import exceptions
class NetworkSegmentRangeNotFound(exceptions.NotFound):
    message = _('Network Segment Range %(range_id)s could not be found.')