import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class MACAddressFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(MACAddressFieldTest, self).setUp()
        self.field = common_types.MACAddressField()
        mac1 = tools.get_random_EUI()
        mac2 = tools.get_random_EUI()
        self.coerce_good_values = [(mac1, mac1), (mac2, mac2)]
        self.coerce_bad_values = ['XXXX', 'ypp', 'g3:vvv', net.get_random_mac('fe:16:3e:00:00:00'.split(':'))]
        self.to_primitive_values = ((a1, str(a2)) for a1, a2 in self.coerce_good_values)
        self.from_primitive_values = ((a2, a1) for a1, a2 in self.to_primitive_values)

    def test_stringify(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual('%s' % in_val, self.field.stringify(in_val))