from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@ddt.data({'iov_queues_requested': 0}, {'offloaded_sa': 0})
@ddt.unpack
def test_set_vswitch_port_offload_invalid(self, iov_queues_requested=1, offloaded_sa=1024):
    self.assertRaises(exceptions.InvalidParameterValue, self.netutils.set_vswitch_port_offload, mock.sentinel.port_name, iov_queues_requested=iov_queues_requested, offloaded_sa=offloaded_sa)