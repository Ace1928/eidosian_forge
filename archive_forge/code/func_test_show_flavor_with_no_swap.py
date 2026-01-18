from novaclient.tests.functional import base
def test_show_flavor_with_no_swap(self):
    _, flv_name = self._create_flavor()
    out = self.nova('flavor-show %s' % flv_name)
    self.assertEqual(self.SWAP_DEFAULT, self._get_value_from_the_table(out, 'swap'))