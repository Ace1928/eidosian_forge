from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_token_credential_with_context(self):
    token_value = 'ec9799cd921e4e0a8ab6111c08ebf065'
    ctxt = context.RequestContext(auth_token=token_value)
    self.config_fixture.config(auth_type='token', group='key_manager')
    token_context = utils.credential_factory(conf=CONF, context=ctxt)
    token_context_class = token_context.__class__.__name__
    self.assertEqual('Token', token_context_class)
    self.assertEqual(token_value, token_context.token)