import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class V3Oauth2ClientCredentialTests(utils.TestCase):

    def setUp(self):
        super(V3Oauth2ClientCredentialTests, self).setUp()
        self.auth_url = uuid.uuid4().hex

    def create(self, **kwargs):
        kwargs.setdefault('auth_url', self.auth_url)
        loader = loading.get_plugin_loader('v3oauth2clientcredential')
        return loader.load_from_options(**kwargs)

    def test_basic(self):
        id = uuid.uuid4().hex
        secret = uuid.uuid4().hex
        oauth2_endpoint = 'https://localhost/token'
        client_cred = self.create(oauth2_endpoint=oauth2_endpoint, oauth2_client_id=id, oauth2_client_secret=secret)
        client_method = client_cred.auth_methods[0]
        self.assertEqual(id, client_method.oauth2_client_id)
        self.assertEqual(secret, client_method.oauth2_client_secret)
        self.assertEqual(oauth2_endpoint, client_method.oauth2_endpoint)
        self.assertEqual(id, client_cred._oauth2_client_id)
        self.assertEqual(secret, client_cred._oauth2_client_secret)
        self.assertEqual(oauth2_endpoint, client_cred._oauth2_endpoint)

    def test_without_oauth2_endpoint(self):
        id = uuid.uuid4().hex
        secret = uuid.uuid4().hex
        self.assertRaises(exceptions.OptionError, self.create, oauth2_client_id=id, oauth2_client_secret=secret)

    def test_without_client_id(self):
        oauth2_endpoint = 'https://localhost/token'
        self.assertRaises(exceptions.OptionError, self.create, oauth2_endpoint=oauth2_endpoint, oauth2_client_secret=uuid.uuid4().hex)

    def test_without_secret(self):
        oauth2_endpoint = 'https://localhost/token'
        self.assertRaises(exceptions.OptionError, self.create, oauth2_endpoint=oauth2_endpoint, oauth2_client_id=uuid.uuid4().hex)