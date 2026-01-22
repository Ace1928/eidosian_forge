from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QosPlacementAllocationUpdateConflict(e.Conflict):
    message = _('Updating placement allocation with %(alloc_diff)s for consumer %(consumer)s failed. The requested resources would exceed the capacity available.')