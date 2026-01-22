from neutron_lib._i18n import _
from neutron_lib import exceptions
class MeteringLabelRuleOverlaps(exceptions.Conflict):
    message = _("Metering label rule with remote_ip_prefix '%(remote_ip_prefix)s' overlaps another.")