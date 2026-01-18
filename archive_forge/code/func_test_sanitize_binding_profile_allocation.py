from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_sanitize_binding_profile_allocation(self):
    old_format = self.RP_UUID
    new_format = {self.GROUP_UUID: self.RP_UUID}
    min_bw_rules = [mock.MagicMock(id=self.MIN_BW_RULE_ID)]
    self.assertEqual(new_format, converters.convert_to_sanitized_binding_profile_allocation(old_format, self.PORT_ID, min_bw_rules))
    self.assertEqual(new_format, converters.convert_to_sanitized_binding_profile_allocation(new_format, self.PORT_ID, min_bw_rules))