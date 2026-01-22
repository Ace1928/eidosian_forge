import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalConductorShow(TestBaremetalConductor):

    def setUp(self):
        super(TestBaremetalConductorShow, self).setUp()
        self.baremetal_mock.conductor.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.CONDUCTOR), loaded=True)
        self.cmd = baremetal_conductor.ShowBaremetalConductor(self.app, None)

    def test_conductor_show(self):
        arglist = ['xxxx.xxxx']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['xxxx.xxxx']
        self.baremetal_mock.conductor.get.assert_called_with(*args, fields=None)
        collist = ('alive', 'conductor_group', 'drivers', 'hostname')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_alive, baremetal_fakes.baremetal_conductor_group, baremetal_fakes.baremetal_drivers, baremetal_fakes.baremetal_hostname)
        self.assertEqual(datalist, tuple(data))

    def test_conductor_show_no_conductor(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_conductor_show_fields(self):
        arglist = ['xxxxx', '--fields', 'hostname', 'alive']
        verifylist = [('conductor', 'xxxxx'), ('fields', [['hostname', 'alive']])]
        fake_cond = copy.deepcopy(baremetal_fakes.CONDUCTOR)
        fake_cond.pop('conductor_group')
        fake_cond.pop('drivers')
        self.baremetal_mock.conductor.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_cond, loaded=True)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('conductor_group', columns)
        args = ['xxxxx']
        fields = ['hostname', 'alive']
        self.baremetal_mock.conductor.get.assert_called_with(*args, fields=fields)

    def test_conductor_show_fields_multiple(self):
        arglist = ['xxxxx', '--fields', 'hostname', 'alive', '--fields', 'conductor_group']
        verifylist = [('conductor', 'xxxxx'), ('fields', [['hostname', 'alive'], ['conductor_group']])]
        fake_cond = copy.deepcopy(baremetal_fakes.CONDUCTOR)
        fake_cond.pop('drivers')
        self.baremetal_mock.conductor.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_cond, loaded=True)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('drivers', columns)
        args = ['xxxxx']
        fields = ['hostname', 'alive', 'conductor_group']
        self.baremetal_mock.conductor.get.assert_called_with(*args, fields=fields)

    def test_conductor_show_invalid_fields(self):
        arglist = ['xxxxx', '--fields', 'hostname', 'invalid']
        verifylist = [('conductor', 'xxxxx'), ('fields', [['hostname', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)