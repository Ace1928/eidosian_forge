from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementResourceProviderGenerationConflict(exceptions.Conflict):
    message = _('Placement resource provider generation does not match with the server side for resource provider: %(resource_provider)s with generation %(generation)s.')