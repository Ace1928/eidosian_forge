import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_v4_creds_false(self):
    credentials = {'host': '127.0.0.1', 'verb': 'GET', 'path': '/v1/', 'params': {'SignatureVersion': '0', 'AWSAccessKeyId': self.access, 'Timestamp': '2012-11-27T11:47:02Z', 'Action': 'Foo'}}
    self.assertFalse(self.signer._v4_creds(credentials))