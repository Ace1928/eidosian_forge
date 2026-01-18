from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_token_credential_exception(self):
    self.config_fixture.config(auth_type='token', group='key_manager')
    self.assertRaises(exception.InsufficientCredentialDataError, utils.credential_factory, CONF)