from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowL7Rule(LbaasL7RuleMixin, neutronV20.ShowCommand):
    """LBaaS v2 Show information of a given rule."""
    resource = 'rule'
    shadow_resource = 'lbaas_l7rule'