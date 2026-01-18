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
def test_validate_route_cidr(self):
    cidr = '10.0.0.0/8'
    msg = validators.validate_route_cidr(cidr, None)
    self.assertIsNone(msg)
    cidr = '192.168.1.1/32'
    msg = validators.validate_route_cidr(cidr, None)
    self.assertIsNone(msg)
    cidr = '192.168.1.1/8'
    msg = validators.validate_route_cidr(cidr, None)
    error = _("'%(data)s' is not a recognized CIDR, '%(cidr)s' is recommended") % {'data': cidr, 'cidr': '192.0.0.0/8'}
    self.assertEqual(error, msg)
    cidr = '127.0.0.0/8'
    msg = validators.validate_route_cidr(cidr, None)
    error = _("'%(data)s' is not a routable CIDR") % {'data': cidr}
    self.assertEqual(error, msg)
    cidr = 'invalid'
    msg = validators.validate_route_cidr(cidr, None)
    error = "'%s' is not a valid CIDR" % cidr
    self.assertEqual(error, msg)