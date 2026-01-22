from neutron_lib._i18n import _
from neutron_lib import exceptions
class NetworkSegmentRangeDefaultReadOnly(exceptions.BadRequest):
    message = _('Network Segment Range %(range_id)s is a default segment range which could not be updated or deleted.')