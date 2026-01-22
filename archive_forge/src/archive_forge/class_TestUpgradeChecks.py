import glance_store
from oslo_config import cfg
from oslo_upgradecheck import upgradecheck
from glance.cmd.status import Checks
from glance.tests import utils as test_utils
class TestUpgradeChecks(test_utils.BaseTestCase):

    def setUp(self):
        super(TestUpgradeChecks, self).setUp()
        glance_store.register_opts(CONF)
        self.checker = Checks()

    def test_sheepdog_removal_no_config(self):
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)

    def test_sheepdog_removal_enabled_backends(self):
        self.config(enabled_backends=None)
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
        self.config(enabled_backends={})
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
        self.config(enabled_backends={'foo': 'bar'})
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
        self.config(enabled_backends={'sheepdog': 'foobar'})
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.FAILURE)

    def test_sheepdog_removal_glance_store_stores(self):
        self.config(stores=None, group='glance_store')
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
        self.config(stores='', group='glance_store')
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
        self.config(stores='foo', group='glance_store')
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.SUCCESS)
        self.config(stores='sheepdog', group='glance_store')
        self.assertEqual(self.checker._check_sheepdog_store().code, upgradecheck.Code.FAILURE)

    def test_owner_is_tenant_removal(self):
        self.config(owner_is_tenant=True)
        self.assertEqual(self.checker._check_owner_is_tenant().code, upgradecheck.Code.SUCCESS)
        self.config(owner_is_tenant=False)
        self.assertEqual(self.checker._check_owner_is_tenant().code, upgradecheck.Code.FAILURE)