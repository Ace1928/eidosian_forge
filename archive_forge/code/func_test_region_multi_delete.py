from openstackclient.tests.functional.identity.v3 import common
def test_region_multi_delete(self):
    region_1 = self._create_dummy_region(add_clean_up=False)
    region_2 = self._create_dummy_region(add_clean_up=False)
    raw_output = self.openstack('region delete %s %s' % (region_1, region_2))
    self.assertEqual(0, len(raw_output))