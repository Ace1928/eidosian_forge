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
def test_validate_integer(self):
    msg = validators.validate_integer(1)
    self.assertIsNone(msg)
    msg = validators.validate_integer(0.1)
    self.assertEqual("'0.1' is not an integer", msg)
    msg = validators.validate_integer('1')
    self.assertIsNone(msg)
    msg = validators.validate_integer('0.1')
    self.assertEqual("'0.1' is not an integer", msg)
    msg = validators.validate_integer(True)
    self.assertEqual("'True' is not an integer:boolean", msg)
    msg = validators.validate_integer(False)
    self.assertEqual("'False' is not an integer:boolean", msg)
    msg = validators.validate_integer(float('Inf'))
    self.assertEqual("'inf' is not an integer", msg)
    msg = validators.validate_integer(None)
    self.assertEqual("'None' is not an integer", msg)