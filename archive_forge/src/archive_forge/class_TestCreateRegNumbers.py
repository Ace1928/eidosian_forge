from neutron_lib.agent.common import constants
from neutron_lib.agent.common import utils
from neutron_lib.tests import _base
class TestCreateRegNumbers(_base.BaseTestCase):

    def test_no_registers_defined(self):
        flow = {'foo': 'bar'}
        utils.create_reg_numbers(flow)
        self.assertEqual({'foo': 'bar'}, flow)

    def test_all_registers_defined(self):
        flow = {'foo': 'bar', constants.PORT_REG_NAME: 1, constants.NET_REG_NAME: 2, constants.REMOTE_GROUP_REG_NAME: 3, constants.INGRESS_BW_LIMIT_REG_NAME: 4, constants.MIN_BW_REG_NAME: 5}
        expected_flow = {'foo': 'bar', 'reg{:d}'.format(constants.REG_PORT): 1, 'reg{:d}'.format(constants.REG_NET): 2, 'reg{:d}'.format(constants.REG_REMOTE_GROUP): 3, 'reg{:d}'.format(constants.REG_INGRESS_BW_LIMIT): 4, 'reg{:d}'.format(constants.REG_MIN_BW): 5}
        utils.create_reg_numbers(flow)
        self.assertEqual(expected_flow, flow)