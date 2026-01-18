import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_subnet_mismatch_for_port(self):
    self._check_nexc(ne.SubnetMismatchForPort, _('Subnet on port porter does not match the requested subnet submit.'), port_id='porter', subnet_id='submit')