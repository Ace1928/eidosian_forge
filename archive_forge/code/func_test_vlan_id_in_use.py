import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_vlan_id_in_use(self):
    self._check_nexc(ne.VlanIdInUse, _('Unable to create the network. The VLAN virtual on physical network phys is in use.'), vlan_id='virtual', physical_network='phys')