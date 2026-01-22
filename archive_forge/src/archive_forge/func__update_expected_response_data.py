from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def _update_expected_response_data(self, data):
    ppg = fakes.FakeSfcPortPairGroup.create_port_pair_group(data)
    self.network.create_sfc_port_pair_group.return_value = ppg
    return self.get_data(ppg)