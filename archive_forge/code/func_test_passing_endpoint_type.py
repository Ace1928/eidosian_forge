from keystoneauth1 import session
from oslo_utils import uuidutils
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.v2 import client
def test_passing_endpoint_type(self):
    endpoint_type = uuidutils.generate_uuid(dashed=False)
    s = session.Session()
    c = client.Client(session=s, endpoint_type=endpoint_type, direct_use=False)
    self.assertEqual(endpoint_type, c.client.interface)