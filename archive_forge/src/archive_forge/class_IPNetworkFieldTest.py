import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class IPNetworkFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(IPNetworkFieldTest, self).setUp()
        self.field = common_types.IPNetworkField()
        addrs = [tools.get_random_ip_network(version=ip_version) for ip_version in const.IP_ALLOWED_VERSIONS]
        self.coerce_good_values = [(addr, addr) for addr in addrs]
        self.coerce_bad_values = ['ypp', 'g3:vvv', '10.0.0.0/24']
        self.to_primitive_values = ((a1, str(a2)) for a1, a2 in self.coerce_good_values)
        self.from_primitive_values = ((a2, a1) for a1, a2 in self.to_primitive_values)

    def test_stringify(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual('%s' % in_val, self.field.stringify(in_val))