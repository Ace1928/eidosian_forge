from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_token_credential(self):
    token_value = 'ec9799cd921e4e0a8ab6111c08ebf065'
    self.config_fixture.config(auth_type='token', token=token_value, group='key_manager')
    token_context = utils.credential_factory(conf=CONF)
    token_context_class = token_context.__class__.__name__
    self.assertEqual('Token', token_context_class)
    self.assertEqual(token_value, token_context.token)