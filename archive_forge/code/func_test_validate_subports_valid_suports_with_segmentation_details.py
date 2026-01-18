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
def test_validate_subports_valid_suports_with_segmentation_details(self):
    body = [{'port_id': '00000000-ffff-ffff-ffff-000000000000', 'segmentation_id': '3', 'segmentation_type': 'vlan'}, {'port_id': '11111111-ffff-ffff-ffff-000000000000', 'segmentation_id': '5', 'segmentation_type': 'vlan'}]
    self.assertIsNone(validators.validate_subports(body))