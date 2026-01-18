import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_versioned_endpoint_no_version(self):
    gc = client.Client(endpoint='http://example.com/v2')
    self.assertEqual('http://example.com', gc.http_client.endpoint)
    self.assertIsInstance(gc, v2.client.Client)