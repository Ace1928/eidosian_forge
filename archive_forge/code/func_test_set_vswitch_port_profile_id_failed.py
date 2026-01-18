from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_set_vswitch_port_profile_id_failed(self):
    self._test_set_vswitch_port_profile_id(found=False, side_effect=Exception)