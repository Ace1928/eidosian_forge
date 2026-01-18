from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_create_default_setting_data')
def test_prepare_profile_sd_failed(self, mock_create_default_sd):
    self.assertRaises(TypeError, self.netutils._prepare_profile_sd, invalid_argument=mock.sentinel.invalid_argument)