from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementAPIVersionIncorrect(exceptions.NotFound):
    message = _('Placement API version %(current_version)s, do not meet the needed version %(needed_version)s.')