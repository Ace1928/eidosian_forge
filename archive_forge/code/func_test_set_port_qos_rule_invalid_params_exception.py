from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@ddt.data({'min_kbps': 100}, {'min_kbps': 10 * units.Ki, 'max_kbps': 100}, {'max_kbps': 10 * units.Ki, 'max_burst_kbps': 100})
def test_set_port_qos_rule_invalid_params_exception(self, qos_rule):
    self.assertRaises(exceptions.InvalidParameterValue, self.netutils.set_port_qos_rule, mock.sentinel.port_id, qos_rule)