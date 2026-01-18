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
def test_success_adding_duplicate_validator(self):
    validators.add_validator('dummy', dummy_validator)
    validators.add_validator('dummy', dummy_validator)
    self.assertEqual(dummy_validator, validators.get_validator('dummy'))