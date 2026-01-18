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
def test_validate_integer_values(self):
    msg = validators.validate_integer(2, [2, 3, 4, 5])
    self.assertIsNone(msg)
    msg = validators.validate_integer(1, [2, 3, 4, 5])
    self.assertEqual('1 is not in valid_values', msg)