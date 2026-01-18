import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_regions(self):
    expected_regions = []
    for _ in range(2):
        region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
        expected_regions.append(region['id'])
    with self.test_client() as c:
        r = c.get('/v3/regions', headers=self.headers)
        for region in r.json['regions']:
            self.assertIn(region['id'], expected_regions)