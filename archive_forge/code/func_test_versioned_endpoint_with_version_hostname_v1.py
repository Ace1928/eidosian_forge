import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_versioned_endpoint_with_version_hostname_v1(self):
    gc = client.Client(endpoint='http://v2.example.com/v1')
    self.assertEqual('http://v2.example.com', gc.http_client.endpoint)
    self.assertIsInstance(gc, v1.client.Client)