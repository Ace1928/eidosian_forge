from novaclient.tests.functional.v2.legacy import test_flavor_access
class TestFlvAccessNovaClientV27(test_flavor_access.TestFlvAccessNovaClient):
    """Check that an attempt to grant an access to a public flavor
      for the given tenant fails with Conflict error in accordance with
      2.7 microversion REST API History
    """
    COMPUTE_API_VERSION = '2.7'

    def test_add_access_public_flavor(self):
        flv_name = self.name_generate()
        self.nova('flavor-create %s auto 512 1 1' % flv_name)
        self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
        output = self.nova('flavor-access-add %s %s' % (flv_name, self.project_id), fail_ok=True, merge_stderr=True)
        self.assertIn('ERROR (Conflict): Can not add access to a public flavor. (HTTP 409) ', output)