from neutron_lib._i18n import _
from neutron_lib import exceptions
class MeteringLabelNotFound(exceptions.NotFound):
    message = _("Metering label '%(label_id)s' does not exist.")