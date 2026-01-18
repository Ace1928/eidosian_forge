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
def test_validate_uuid_list(self):
    bad_uuid_list = ['00000000-ffff-ffff-ffff-000000000000', '00000000-ffff-ffff-ffff-000000000001', '123']
    msg = validators.validate_uuid_list(bad_uuid_list, valid_values='parameter not used')
    error = "'%s' is not a valid UUID" % bad_uuid_list[2]
    self.assertEqual(error, msg)
    good_uuid_list = ['00000000-ffff-ffff-ffff-000000000000', '00000000-ffff-ffff-ffff-000000000001']
    msg = validators.validate_uuid_list(good_uuid_list, valid_values='parameter not used')
    self.assertIsNone(msg)