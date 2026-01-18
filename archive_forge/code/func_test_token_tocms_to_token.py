import errno
import os
import subprocess
from unittest import mock
import testresources
from testtools import matchers
from keystoneclient.common import cms
from keystoneclient import exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
def test_token_tocms_to_token(self):
    with open(os.path.join(client_fixtures.CMSDIR, 'auth_token_scoped.pem')) as f:
        AUTH_TOKEN_SCOPED_CMS = f.read()
    self.assertEqual(cms.token_to_cms(self.examples.SIGNED_TOKEN_SCOPED), AUTH_TOKEN_SCOPED_CMS)
    tok = cms.cms_to_token(cms.token_to_cms(self.examples.SIGNED_TOKEN_SCOPED))
    self.assertEqual(tok, self.examples.SIGNED_TOKEN_SCOPED)