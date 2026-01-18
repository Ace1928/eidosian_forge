import datetime
import hashlib
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone.common import utils
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_valid_authentication_response_with_proper_secret(self):
    signer = ec2_utils.Ec2Signer(self.cred_blob['secret'])
    timestamp = utils.isotime(timeutils.utcnow())
    credentials = {'access': self.cred_blob['access'], 'secret': self.cred_blob['secret'], 'host': 'localhost', 'verb': 'GET', 'path': '/', 'params': {'SignatureVersion': '2', 'Action': 'Test', 'Timestamp': timestamp}}
    credentials['signature'] = signer.generate(credentials)
    resp = self.post('/ec2tokens', body={'credentials': credentials}, expected_status=http.client.OK)
    self.assertValidProjectScopedTokenResponse(resp, self.user)