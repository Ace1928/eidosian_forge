import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class DomainNameFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(DomainNameFieldTest, self).setUp()
        self.field = common_types.DomainNameField()
        self.coerce_good_values = [(val, val) for val in ('www.google.com', 'hostname', '1abc.com')]
        self.coerce_bad_values = ['x' * (db_const.FQDN_FIELD_SIZE + 1), 10, []]
        self.to_primitive_values = self.coerce_good_values
        self.from_primitive_values = self.coerce_good_values

    def test_stringify(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual("'%s'" % in_val, self.field.stringify(in_val))