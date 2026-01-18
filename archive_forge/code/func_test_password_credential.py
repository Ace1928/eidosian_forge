from castellan.common import exception
from castellan.common import utils
from castellan.tests import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
def test_password_credential(self):
    password_value = 'p4ssw0rd'
    self.config_fixture.config(auth_type='password', password=password_value, group='key_manager')
    password_context = utils.credential_factory(conf=CONF)
    password_context_class = password_context.__class__.__name__
    self.assertEqual('Password', password_context_class)
    self.assertEqual(password_value, password_context.password)