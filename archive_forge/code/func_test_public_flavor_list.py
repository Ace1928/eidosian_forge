from novaclient.tests.functional import base
def test_public_flavor_list(self):
    flavor_list1 = self.nova('flavor-list')
    flavor_list2 = self.another_nova('flavor-list')
    self.assertEqual(flavor_list1, flavor_list2)