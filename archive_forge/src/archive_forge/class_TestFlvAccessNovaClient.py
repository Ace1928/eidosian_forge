from novaclient.tests.functional import base
class TestFlvAccessNovaClient(base.TenantTestBase):
    """Functional tests for flavors with public and non-public access"""
    COMPUTE_API_VERSION = '2.1'

    def test_public_flavor_list(self):
        flavor_list1 = self.nova('flavor-list')
        flavor_list2 = self.another_nova('flavor-list')
        self.assertEqual(flavor_list1, flavor_list2)

    def test_non_public_flavor_list(self):
        flv_name = self.name_generate()
        self.nova('flavor-create --is-public false %s auto 512 1 1' % flv_name)
        self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
        flavor_list1 = self.nova('flavor-list')
        self.assertNotIn(flv_name, flavor_list1)
        flavor_list2 = self.nova('flavor-list --all')
        flavor_list3 = self.another_nova('flavor-list --all')
        self.assertIn(flv_name, flavor_list2)
        self.assertNotIn(flv_name, flavor_list3)

    def test_add_access_non_public_flavor(self):
        flv_name = self.name_generate()
        self.nova('flavor-create --is-public false %s auto 512 1 1' % flv_name)
        self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
        self.nova('flavor-access-add', params='%s %s' % (flv_name, self.project_id))
        self.assertIn(self.project_id, self.nova('flavor-access-list --flavor %s' % flv_name))

    def test_add_access_public_flavor(self):
        flv_name = self.name_generate()
        self.nova('flavor-create %s auto 512 1 1' % flv_name)
        self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
        self.nova('flavor-access-add %s %s' % (flv_name, self.project_id))
        output = self.nova('flavor-access-list --flavor %s' % flv_name, fail_ok=True, merge_stderr=True)
        self.assertIn('CommandError', output)