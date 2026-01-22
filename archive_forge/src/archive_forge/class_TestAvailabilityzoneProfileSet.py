import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzoneprofile
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneProfileSet(TestAvailabilityzoneProfile):

    def setUp(self):
        super().setUp()
        self.cmd = availabilityzoneprofile.SetAvailabilityzoneProfile(self.app, None)

    def test_availabilityzoneprofile_set(self):
        arglist = [self._availabilityzoneprofile.id, '--name', 'new_name']
        verifylist = [('availabilityzoneprofile', self._availabilityzoneprofile.id), ('name', 'new_name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzoneprofile_set.assert_called_with(self._availabilityzoneprofile.id, json={'availability_zone_profile': {'name': 'new_name'}})
        arglist = [self._availabilityzoneprofile.id, '--availability-zone-data', '{"key1": "value1"}']
        verifylist = [('availabilityzoneprofile', self._availabilityzoneprofile.id), ('availability_zone_data', '{"key1": "value1"}')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzoneprofile_set.assert_called_with(self._availabilityzoneprofile.id, json={'availability_zone_profile': {'availability_zone_data': '{"key1": "value1"}'}})