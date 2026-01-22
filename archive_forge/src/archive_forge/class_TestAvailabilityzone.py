import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzone(fakes.TestOctaviaClient):

    def setUp(self):
        super().setUp()
        self._availabilityzone = fakes.createFakeResource('availability_zone')
        self.availabilityzone_info = copy.deepcopy(attr_consts.AVAILABILITY_ZONE_ATTRS)
        self.columns = copy.deepcopy(constants.AVAILABILITYZONE_COLUMNS)
        self.api_mock = mock.Mock()
        self.api_mock.availabilityzone_list.return_value = copy.deepcopy({'availability_zones': [attr_consts.AVAILABILITY_ZONE_ATTRS]})
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock