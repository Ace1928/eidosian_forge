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
def test_validate_no_whitespace(self):
    data = 'no_white_space'
    result = validators.validate_no_whitespace(data)
    self.assertEqual(data, result)
    self.assertRaises(n_exc.InvalidInput, validators.validate_no_whitespace, 'i have whitespace')
    self.assertRaises(n_exc.InvalidInput, validators.validate_no_whitespace, 'i\thave\twhitespace')
    for ws in string.whitespace:
        self.assertRaises(n_exc.InvalidInput, validators.validate_no_whitespace, '%swhitespace-at-head' % ws)
        self.assertRaises(n_exc.InvalidInput, validators.validate_no_whitespace, 'whitespace-at-tail%s' % ws)