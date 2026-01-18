import glance_store
from oslo_config import cfg
from oslo_upgradecheck import upgradecheck
from glance.cmd.status import Checks
from glance.tests import utils as test_utils
def test_owner_is_tenant_removal(self):
    self.config(owner_is_tenant=True)
    self.assertEqual(self.checker._check_owner_is_tenant().code, upgradecheck.Code.SUCCESS)
    self.config(owner_is_tenant=False)
    self.assertEqual(self.checker._check_owner_is_tenant().code, upgradecheck.Code.FAILURE)