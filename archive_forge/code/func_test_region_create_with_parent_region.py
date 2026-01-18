from openstackclient.tests.functional.identity.v3 import common
def test_region_create_with_parent_region(self):
    parent_region_id = self._create_dummy_region()
    self._create_dummy_region(parent_region=parent_region_id)