from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementTraitNotFound(exceptions.NotFound):
    message = _('Placement trait not found %(trait)s.')