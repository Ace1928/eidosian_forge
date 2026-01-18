import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
def test_stream_object(self):
    text = b'test body'
    self.register_uris([dict(method='GET', uri=self.object_endpoint, headers={'Content-Length': '20304400896', 'Content-Type': 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Last-Modified': 'Thu, 15 Dec 2016 13:34:14 GMT', 'Etag': '"b5c454b44fbd5344793e3fb7e3850768"', 'X-Timestamp': '1481808853.65009', 'X-Trans-Id': 'tx68c2a2278f0c469bb6de1-005857ed80dfw1', 'Date': 'Mon, 19 Dec 2016 14:24:00 GMT', 'X-Static-Large-Object': 'True', 'X-Object-Meta-Mtime': '1481513709.168512'}, text='test body')])
    response_text = b''
    for data in self.cloud.stream_object(self.container, self.object):
        response_text += data
    self.assert_calls()
    self.assertEqual(text, response_text)