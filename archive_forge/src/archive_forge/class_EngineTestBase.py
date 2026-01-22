import contextlib
import string
import threading
import time
from oslo_utils import timeutils
import redis
from taskflow import exceptions
from taskflow.listeners import capturing
from taskflow.persistence.backends import impl_memory
from taskflow import retry
from taskflow import task
from taskflow.types import failure
from taskflow.utils import kazoo_utils
from taskflow.utils import redis_utils
class EngineTestBase(object):

    def setUp(self):
        super(EngineTestBase, self).setUp()
        self.backend = impl_memory.MemoryBackend(conf={})

    def tearDown(self):
        EngineTestBase.values = None
        with contextlib.closing(self.backend) as be:
            with contextlib.closing(be.get_connection()) as conn:
                conn.clear_all()
        super(EngineTestBase, self).tearDown()

    def _make_engine(self, flow, **kwargs):
        raise exceptions.NotImplementedError('_make_engine() must be overridden if an engine is desired')