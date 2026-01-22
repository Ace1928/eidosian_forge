from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementResourceProviderNameNotUnique(exceptions.Conflict):
    message = _('Another resource provider exists with the provided name: %(name)s.')