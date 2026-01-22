from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementResourceProviderNotFound(exceptions.NotFound):
    message = _('Placement resource provider not found %(resource_provider)s.')