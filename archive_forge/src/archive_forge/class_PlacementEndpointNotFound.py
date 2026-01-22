from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementEndpointNotFound(exceptions.NotFound):
    message = _('Placement API endpoint not found.')