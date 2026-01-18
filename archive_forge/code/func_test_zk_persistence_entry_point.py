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
def test_zk_persistence_entry_point(self):
    conf = {'connection': 'zookeeper:'}
    with contextlib.closing(backends.fetch(conf)) as be:
        self.assertIsInstance(be, impl_zookeeper.ZkBackend)