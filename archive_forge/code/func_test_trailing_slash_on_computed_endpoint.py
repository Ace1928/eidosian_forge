import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_trailing_slash_on_computed_endpoint(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    self.requests_mock.get(self.TEST_VOLUME.unversioned.public + '/', json=disc)
    s.get_endpoint_data(service_type='block-storage', interface='public', min_version='2.0', max_version='2.latest', project_id=self.project_id)
    self.assertTrue(self.requests_mock.request_history[-1].url.endswith('/'))