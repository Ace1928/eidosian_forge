import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneShow(TestAvailabilityzone):

    def setUp(self):
        super().setUp()
        mock_show = self.api_mock.availabilityzone_show
        mock_show.return_value = self.availabilityzone_info
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = availabilityzone.ShowAvailabilityzone(self.app, None)

    def test_availabilityzone_show(self):
        arglist = [self._availabilityzone.name]
        verifylist = [('availabilityzone', self._availabilityzone.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzone_show.assert_called_with(availabilityzone_name=self._availabilityzone.name)