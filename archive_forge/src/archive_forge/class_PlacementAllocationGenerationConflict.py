from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementAllocationGenerationConflict(exceptions.Conflict):
    message = _('Resource allocation has been changed for consumer %(consumer)s in Placement while Neutron tried to update it.')