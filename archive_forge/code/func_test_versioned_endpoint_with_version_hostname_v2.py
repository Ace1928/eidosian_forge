import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_versioned_endpoint_with_version_hostname_v2(self):
    gc = client.Client(endpoint='http://v1.example.com/v2')
    self.assertEqual('http://v1.example.com', gc.http_client.endpoint)
    self.assertIsInstance(gc, v2.client.Client)