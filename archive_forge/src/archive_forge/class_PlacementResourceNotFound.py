from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementResourceNotFound(exceptions.NotFound):
    message = _('Placement resource not found on url: %(url)s.')