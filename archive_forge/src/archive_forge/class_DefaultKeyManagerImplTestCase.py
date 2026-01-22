from oslo_config import cfg
from oslo_config import fixture
from castellan import key_manager
from castellan.key_manager import barbican_key_manager
from castellan.tests import base
class DefaultKeyManagerImplTestCase(KeyManagerTestCase):

    def _create_key_manager(self):
        return key_manager.API(self.conf)

    def test_default_key_manager(self):
        self.assertEqual('barbican', self.conf.key_manager.backend)
        self.assertIsNotNone(self.key_mgr)
        self.assertIsInstance(self.key_mgr, barbican_key_manager.BarbicanKeyManager)