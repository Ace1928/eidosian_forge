import glance_store
from oslo_config import cfg
from oslo_upgradecheck import upgradecheck
from glance.cmd.status import Checks
from glance.tests import utils as test_utils
def test_sheepdog_removal_enabled_backends(self):
    self.config(enabled_backends=None)
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
    self.config(enabled_backends={})
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
    self.config(enabled_backends={'foo': 'bar'})
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
    self.config(enabled_backends={'sheepdog': 'foobar'})
    self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.FAILURE)