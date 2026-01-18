import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
@mock.patch('neutron_lib.api.validators.validate_range')
def test_validate_range_or_none(self, mock_validate_range):
    msg = validators.validate_range_or_none(None, [1, 9])
    self.assertFalse(mock_validate_range.called)
    self.assertIsNone(msg)
    validators.validate_range_or_none(1, [1, 9])
    mock_validate_range.assert_called_once_with(1, [1, 9])