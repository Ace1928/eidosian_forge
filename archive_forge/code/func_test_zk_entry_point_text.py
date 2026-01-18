import contextlib
from zake import fake_client
from taskflow.jobs import backends
from taskflow.jobs.backends import impl_redis
from taskflow.jobs.backends import impl_zookeeper
from taskflow import test
def test_zk_entry_point_text(self):
    conf = 'zookeeper'
    with contextlib.closing(backends.fetch('test', conf)) as be:
        self.assertIsInstance(be, impl_zookeeper.ZookeeperJobBoard)