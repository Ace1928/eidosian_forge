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
def _test_validate_regex(self, validator, allow_none=False):
    pattern = '[hc]at'
    data = None
    msg = validator(data, pattern)
    if allow_none:
        self.assertIsNone(msg)
    else:
        self.assertEqual("'None' is not a valid input", msg)
    data = 'bat'
    msg = validator(data, pattern)
    self.assertEqual("'%s' is not a valid input" % data, msg)
    data = 'hat'
    msg = validator(data, pattern)
    self.assertIsNone(msg)
    data = 'cat'
    msg = validator(data, pattern)
    self.assertIsNone(msg)