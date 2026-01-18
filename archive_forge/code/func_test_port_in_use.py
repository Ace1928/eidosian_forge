import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_port_in_use(self):
    self._check_nexc(ne.PortInUse, _('Unable to complete operation on port a for network c. Port already has an attached device b.'), port_id='a', device_id='b', net_id='c')