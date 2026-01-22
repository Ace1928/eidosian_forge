from novaclient.tests.unit.fixture_data import availability_zones as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import availability_zones
def _get_availability_zone_type(self):
    return availability_zones.AvailabilityZone