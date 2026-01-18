import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_tunnel_id_in_use(self):
    self._check_nexc(ne.TunnelIdInUse, _('Unable to create the network. The tunnel ID sewer is in use.'), tunnel_id='sewer')