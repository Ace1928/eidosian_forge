from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_max_burst_kbits_to_zero(self):
    self._set_max_burst_kbits(max_burst_kbits=0)