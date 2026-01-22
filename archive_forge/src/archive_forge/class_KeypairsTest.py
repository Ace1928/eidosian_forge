from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import keypairs as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import keypairs
class KeypairsTest(utils.FixturedTestCase):
    client_fixture_class = client.V1
    data_fixture_class = data.V1

    def setUp(self):
        super(KeypairsTest, self).setUp()
        self.keypair_type = self._get_keypair_type()
        self.keypair_prefix = self._get_keypair_prefix()

    def _get_keypair_type(self):
        return keypairs.Keypair

    def _get_keypair_prefix(self):
        return keypairs.KeypairManager.keypair_prefix

    def test_get_keypair(self):
        kp = self.cs.keypairs.get('test')
        self.assert_request_id(kp, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/%s/test' % self.keypair_prefix)
        self.assertIsInstance(kp, keypairs.Keypair)
        self.assertEqual('test', kp.name)

    def test_list_keypairs(self):
        kps = self.cs.keypairs.list()
        self.assert_request_id(kps, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/%s' % self.keypair_prefix)
        for kp in kps:
            self.assertIsInstance(kp, keypairs.Keypair)

    def test_delete_keypair(self):
        kp = self.cs.keypairs.list()[0]
        ret = kp.delete()
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('DELETE', '/%s/test' % self.keypair_prefix)
        ret = self.cs.keypairs.delete('test')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('DELETE', '/%s/test' % self.keypair_prefix)
        ret = self.cs.keypairs.delete(kp)
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('DELETE', '/%s/test' % self.keypair_prefix)