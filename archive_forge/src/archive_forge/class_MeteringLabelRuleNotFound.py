from neutron_lib._i18n import _
from neutron_lib import exceptions
class MeteringLabelRuleNotFound(exceptions.NotFound):
    message = _("Metering label rule '%(rule_id)s' does not exist.")