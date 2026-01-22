from osc_lib import utils as oscutils
from manilaclient.osc.v2 import availability_zones as osc_availability_zones
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestAvailabilityZones(manila_fakes.TestShare):

    def setUp(self):
        super(TestAvailabilityZones, self).setUp()
        self.zones_mock = self.app.client_manager.share.availability_zones
        self.zones_mock.reset_mock()