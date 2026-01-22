from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementClientError(exceptions.NeutronException):
    message = _('Placement Client Error (4xx): %(msg)s')