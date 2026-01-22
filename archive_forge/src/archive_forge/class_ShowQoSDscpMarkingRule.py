from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class ShowQoSDscpMarkingRule(qos_rule.QosRuleMixin, neutronv20.ShowCommand):
    """Show information about the given qos dscp marking rule."""
    resource = DSCP_MARKING_RESOURCE
    allow_names = False