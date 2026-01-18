import json
from unittest import mock
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session
from heat.common import auth_plugin
from heat.common import config
from heat.common import exception
from heat.common import policy
from heat.tests import common
def test_parse_auth_credential_to_dict(self):
    cred_dict = json.loads(self.credential)
    self.assertEqual(cred_dict, auth_plugin.parse_auth_credential_to_dict(self.credential))