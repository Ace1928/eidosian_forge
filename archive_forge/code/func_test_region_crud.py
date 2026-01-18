import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_region_crud(self):
    region_id = 'default'
    new_region = unit.new_region_ref(id=region_id)
    res = PROVIDERS.catalog_api.create_region(new_region)
    expected_region = new_region.copy()
    expected_region['parent_region_id'] = None
    self.assertDictEqual(expected_region, res)
    parent_region_id = region_id
    new_region = unit.new_region_ref(parent_region_id=parent_region_id)
    region_id = new_region['id']
    res = PROVIDERS.catalog_api.create_region(new_region)
    self.assertDictEqual(new_region, res)
    regions = PROVIDERS.catalog_api.list_regions()
    self.assertThat(regions, matchers.HasLength(2))
    region_ids = [x['id'] for x in regions]
    self.assertIn(parent_region_id, region_ids)
    self.assertIn(region_id, region_ids)
    region_desc_update = {'description': uuid.uuid4().hex}
    res = PROVIDERS.catalog_api.update_region(region_id, region_desc_update)
    expected_region = new_region.copy()
    expected_region['description'] = region_desc_update['description']
    self.assertDictEqual(expected_region, res)
    PROVIDERS.catalog_api.delete_region(parent_region_id)
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.delete_region, parent_region_id)
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, parent_region_id)
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_id)