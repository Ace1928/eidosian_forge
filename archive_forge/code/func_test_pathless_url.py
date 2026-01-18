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
def test_pathless_url(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    url = 'http://path.less.url:1234'
    disc.add_microversion(href=url, id='v2.1')
    self.stub_url('GET', base_url=url, status_code=200, json=disc)
    token = fixture.V2Token()
    service = token.add_service('network')
    service.add_endpoint(public=url, admin=url, internal=url)
    self.stub_url('POST', ['tokens'], base_url=url, json=token)
    v2_auth = identity.V2Password(url, username='u', password='p')
    sess = session.Session(auth=v2_auth)
    data = sess.get_endpoint_data(service_type='network')
    self.assertEqual(url, data.url)
    self.assertEqual((2, 1), data.api_version)
    self.assertEqual(3, len(list(data._get_discovery_url_choices(project_id='42'))))