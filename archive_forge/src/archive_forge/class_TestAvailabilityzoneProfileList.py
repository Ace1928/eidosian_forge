import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzoneprofile
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneProfileList(TestAvailabilityzoneProfile):

    def setUp(self):
        super().setUp()
        self.datalist = (tuple((attr_consts.AVAILABILITY_ZONE_PROFILE_ATTRS[k] for k in self.columns)),)
        self.cmd = availabilityzoneprofile.ListAvailabilityzoneProfile(self.app, None)

    def test_availabilityzoneprofile_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzoneprofile_list.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_availabilityzoneprofile_list_with_options(self):
        arglist = ['--name', 'availabilityzoneprofile1']
        verifylist = [('name', 'availabilityzoneprofile1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzoneprofile_list.assert_called_with(name='availabilityzoneprofile1')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))