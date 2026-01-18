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
def test_validate_fixed_ips(self):
    fixed_ips = [{'data': [{'subnet_id': '00000000-ffff-ffff-ffff-000000000000', 'ip_address': '1111.1.1.1'}], 'error_msg': "'1111.1.1.1' is not a valid IP address"}, {'data': [{'subnet_id': 'invalid', 'ip_address': '1.1.1.1'}], 'error_msg': "'invalid' is not a valid UUID"}, {'data': None, 'error_msg': "Invalid data format for fixed IP: 'None'"}, {'data': '1.1.1.1', 'error_msg': "Invalid data format for fixed IP: '1.1.1.1'"}, {'data': ['00000000-ffff-ffff-ffff-000000000000', '1.1.1.1'], 'error_msg': "Invalid data format for fixed IP: '00000000-ffff-ffff-ffff-000000000000'"}, {'data': [['00000000-ffff-ffff-ffff-000000000000', '1.1.1.1']], 'error_msg': "Invalid data format for fixed IP: '['00000000-ffff-ffff-ffff-000000000000', '1.1.1.1']'"}, {'data': [{'subnet_id': '00000000-0fff-ffff-ffff-000000000000', 'ip_address': '1.1.1.1'}, {'subnet_id': '00000000-ffff-ffff-ffff-000000000000', 'ip_address': '1.1.1.1'}], 'error_msg': "Duplicate IP address '1.1.1.1'"}]
    for fixed in fixed_ips:
        msg = validators.validate_fixed_ips(fixed['data'])
        self.assertEqual(fixed['error_msg'], msg)
    fixed_ips = [[{'subnet_id': '00000000-ffff-ffff-ffff-000000000000', 'ip_address': '1.1.1.1'}], [{'subnet_id': '00000000-0fff-ffff-ffff-000000000000', 'ip_address': '1.1.1.1'}, {'subnet_id': '00000000-ffff-ffff-ffff-000000000000', 'ip_address': '1.1.1.2'}]]
    for fixed in fixed_ips:
        msg = validators.validate_fixed_ips(fixed)
        self.assertIsNone(msg)