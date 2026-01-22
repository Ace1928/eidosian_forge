from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertToSanitizedBindingProfileAllocation(base.BaseTestCase):
    RP_UUID = '41d7391e-1f69-11ec-a899-8f9d6d950f8d'
    PORT_ID = '64d01804-1f83-11ec-987c-7f6caec3998b'
    MIN_BW_RULE_ID = '52441596-1f83-11ec-93c5-9b759591a493'
    GROUP_UUID = '2a1be6ea-15b0-5ac1-9d70-643e2ae306cb'

    def test_sanitize_binding_profile_allocation(self):
        old_format = self.RP_UUID
        new_format = {self.GROUP_UUID: self.RP_UUID}
        min_bw_rules = [mock.MagicMock(id=self.MIN_BW_RULE_ID)]
        self.assertEqual(new_format, converters.convert_to_sanitized_binding_profile_allocation(old_format, self.PORT_ID, min_bw_rules))
        self.assertEqual(new_format, converters.convert_to_sanitized_binding_profile_allocation(new_format, self.PORT_ID, min_bw_rules))