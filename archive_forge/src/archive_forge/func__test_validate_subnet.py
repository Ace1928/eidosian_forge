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
def _test_validate_subnet(self, validator, allow_none=False):
    cidr = '10.0.2.0/24'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = 'fe80::/24'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = 'fe80::/24'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = 'fe80:0:0:0:0:0:0:0/128'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = '2001:0db8:0:0:1::1/128'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = '2001:0db8::1:0:0:1/128'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = '2001::0:1:0:0:1100/120'
    msg = validator(cidr, None)
    self.assertIsNone(msg)
    cidr = '10/24'
    msg = validator(cidr, None)
    error = _("'%(data)s' isn't a recognized IP subnet cidr, '%(cidr)s' is recommended") % {'data': cidr, 'cidr': '10.0.0.0/24'}
    self.assertEqual(error, msg)
    cidr = '10.0.2.0'
    msg = validator(cidr, None)
    error = _("'%(data)s' isn't a recognized IP subnet cidr, '%(cidr)s' is recommended") % {'data': cidr, 'cidr': '10.0.2.0/32'}
    self.assertEqual(error, msg)
    for i in range(1, 255):
        cidr = '192.168.1.%s/24' % i
        msg = validator(cidr, None)
        self.assertIsNone(msg)
    cidr = 'fe80::'
    msg = validator(cidr, None)
    error = _("'%(data)s' isn't a recognized IP subnet cidr, '%(cidr)s' is recommended") % {'data': cidr, 'cidr': 'fe80::/128'}
    self.assertEqual(error, msg)
    cidr = 'fe80::0'
    msg = validator(cidr, None)
    error = _("'%(data)s' isn't a recognized IP subnet cidr, '%(cidr)s' is recommended") % {'data': cidr, 'cidr': 'fe80::/128'}
    self.assertEqual(error, msg)
    cidr = 'invalid'
    msg = validator(cidr, None)
    error = "'%s' is not a valid IP subnet" % cidr
    self.assertEqual(error, msg)
    cidr = None
    msg = validator(cidr, None)
    if allow_none:
        self.assertIsNone(msg)
    else:
        error = "'%s' is not a valid IP subnet" % cidr
        self.assertEqual(error, msg)
    cidr = '10.0.2.0/24\r'
    msg = validator(cidr, None)
    error = "'%s' is not a valid IP subnet" % cidr
    self.assertEqual(error, msg)