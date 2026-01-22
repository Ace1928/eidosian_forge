import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class CredentialBaseTestCase(test_v3.RestfulTestCase):

    def setUp(self):
        super(CredentialBaseTestCase, self).setUp()
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_fernet.MAX_ACTIVE_KEYS))

    def _create_dict_blob_credential(self):
        blob, credential = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
        credential['blob'] = blob
        credential_id = credential['id']
        PROVIDERS.credential_api.create_credential(credential_id, credential)
        return (json.dumps(blob), credential_id)

    def _test_get_token(self, access, secret):
        """Test signature validation with the access/secret provided."""
        signer = ec2_utils.Ec2Signer(secret)
        params = {'SignatureMethod': 'HmacSHA256', 'SignatureVersion': '2', 'AWSAccessKeyId': access}
        request = {'host': 'foo', 'verb': 'GET', 'path': '/bar', 'params': params}
        signature = signer.generate(request)
        sig_ref = {'access': access, 'signature': signature, 'host': 'foo', 'verb': 'GET', 'path': '/bar', 'params': params}
        r = self.post('/ec2tokens', body={'ec2Credentials': sig_ref}, expected_status=http.client.OK)
        self.assertValidTokenResponse(r)
        return r.result['token']