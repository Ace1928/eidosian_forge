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
def test_validate_list_of_regex_or_none(self):
    pattern = '[hc]at|^$'
    list_of_regex = ['hat', 'cat', '']
    msg = validators.validate_list_of_regex_or_none(list_of_regex, pattern)
    self.assertIsNone(msg)
    list_of_regex = ['bat', 'hat', 'cat', '']
    msg = validators.validate_list_of_regex_or_none(list_of_regex, pattern)
    self.assertEqual("'bat' is not a valid input", msg)
    empty_list = []
    msg = validators.validate_list_of_regex_or_none(empty_list, pattern)