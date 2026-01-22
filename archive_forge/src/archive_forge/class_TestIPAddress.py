import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class TestIPAddress(TestField):

    def setUp(self):
        super(TestIPAddress, self).setUp()
        self.field = fields.IPAddressField()
        self.coerce_good_values = [('1.2.3.4', netaddr.IPAddress('1.2.3.4')), ('::1', netaddr.IPAddress('::1')), (netaddr.IPAddress('::1'), netaddr.IPAddress('::1'))]
        self.coerce_bad_values = ['1-2', 'foo']
        self.to_primitive_values = [(netaddr.IPAddress('1.2.3.4'), '1.2.3.4'), (netaddr.IPAddress('::1'), '::1')]
        self.from_primitive_values = [('1.2.3.4', netaddr.IPAddress('1.2.3.4')), ('::1', netaddr.IPAddress('::1'))]