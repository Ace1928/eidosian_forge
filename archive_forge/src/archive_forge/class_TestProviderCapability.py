import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import provider
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestProviderCapability(fakes.TestOctaviaClient):

    def setUp(self):
        super().setUp()
        self.api_mock = mock.Mock()
        self.api_mock.provider_flavor_capability_list.return_value = copy.deepcopy({'flavor_capabilities': [attr_consts.CAPABILITY_ATTRS]})
        self.api_mock.provider_availability_zone_capability_list.return_value = copy.deepcopy({'availability_zone_capabilities': [attr_consts.CAPABILITY_ATTRS]})
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock