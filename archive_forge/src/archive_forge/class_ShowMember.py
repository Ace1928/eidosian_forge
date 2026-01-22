from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowMember(LbaasMemberMixin, neutronV20.ShowCommand):
    """LBaaS v2 Show information of a given member."""
    resource = 'member'
    shadow_resource = 'lbaas_member'