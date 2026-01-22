from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ShowMeteringLabelRule(neutronv20.ShowCommand):
    """Show information of a given metering label rule."""
    resource = 'metering_label_rule'
    allow_names = False