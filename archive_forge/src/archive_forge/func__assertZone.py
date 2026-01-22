from cinderclient.v3 import availability_zones
from cinderclient.v3 import shell
from cinderclient.tests.unit.fixture_data import availability_zones as azfixture  # noqa
from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit import utils
def _assertZone(self, zone, name, status):
    self.assertEqual(name, zone.zoneName)
    self.assertEqual(status, zone.zoneState)