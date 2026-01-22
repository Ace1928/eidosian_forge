import contextlib
from kazoo import exceptions as kazoo_exceptions
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_zookeeper
from taskflow import test
from taskflow.tests.unit.persistence import base
from taskflow.tests import utils as test_utils
from taskflow.utils import kazoo_utils
@testtools.skipIf(not _ZOOKEEPER_AVAILABLE, 'zookeeper is not available')
class ZkPersistenceTest(test.TestCase, base.PersistenceTestMixin):

    def _get_connection(self):
        return self.backend.get_connection()

    def setUp(self):
        super(ZkPersistenceTest, self).setUp()
        conf = test_utils.ZK_TEST_CONFIG.copy()
        conf['path'] = TEST_PATH_TPL % uuidutils.generate_uuid()
        try:
            self.backend = impl_zookeeper.ZkBackend(conf)
        except Exception as e:
            self.skipTest('Failed creating backend created from configuration %s due to %s' % (conf, e))
        else:
            self.addCleanup(self.backend.close)
            self.addCleanup(clean_backend, self.backend, conf)
            with contextlib.closing(self.backend.get_connection()) as conn:
                conn.upgrade()

    def test_zk_persistence_entry_point(self):
        conf = {'connection': 'zookeeper:'}
        with contextlib.closing(backends.fetch(conf)) as be:
            self.assertIsInstance(be, impl_zookeeper.ZkBackend)