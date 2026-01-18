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
def test_validate_oneline_not_empty_string(self):
    data = 'Test'
    msg = validators.validate_oneline_not_empty_string(data, None)
    self.assertIsNone(msg)
    data = 'Test but this is too long'
    max_len = 4
    msg = validators.validate_oneline_not_empty_string(data, max_len)
    self.assertEqual("'%s' exceeds maximum length of %s" % (data, max_len), msg)
    data = 'First line\nsecond line'
    msg = validators.validate_oneline_not_empty_string(data, None)
    self.assertEqual("Multi-line string is not allowed: '%s'" % data, msg)
    data = ''
    msg = validators.validate_oneline_not_empty_string(data, None)
    self.assertEqual("'%s' Blank strings are not permitted" % data, msg)