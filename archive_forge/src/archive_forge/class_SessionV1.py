import fixtures
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
from novaclient import client
class SessionV1(V1):

    def new_client(self):
        self.session = session.Session()
        loader = loading.get_plugin_loader('password')
        self.session.auth = loader.load_from_options(auth_url=self.identity_url, username='xx', password='xx')
        return client.Client('2', session=self.session)