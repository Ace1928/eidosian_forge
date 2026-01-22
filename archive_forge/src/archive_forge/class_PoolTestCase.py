import threading
import uuid
import fixtures
import testscenarios
from oslo_messaging._drivers import pool
from oslo_messaging.tests import utils as test_utils
class PoolTestCase(test_utils.BaseTestCase):
    _max_size = [('default_size', dict(max_size=None, n_iters=4)), ('set_max_size', dict(max_size=10, n_iters=10))]
    _create_error = [('no_create_error', dict(create_error=False)), ('create_error', dict(create_error=True))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._max_size, cls._create_error)

    class TestPool(pool.Pool):

        def create(self, retry=None):
            return uuid.uuid4()

    class ThreadWaitWaiter(object):
        """A gross hack.

        Stub out the condition variable's wait() method and spin until it
        has been called by each thread.
        """

        def __init__(self, cond, n_threads, test):
            self.cond = cond
            self.test = test
            self.n_threads = n_threads
            self.n_waits = 0
            self.orig_wait = cond.wait

            def count_waits(**kwargs):
                self.n_waits += 1
                self.orig_wait(**kwargs)
            self.test.useFixture(fixtures.MockPatchObject(self.cond, 'wait', count_waits))

        def wait(self):
            while self.n_waits < self.n_threads:
                pass
            self.test.useFixture(fixtures.MockPatchObject(self.cond, 'wait', self.orig_wait))

    def test_pool(self):
        kwargs = {}
        if self.max_size is not None:
            kwargs['max_size'] = self.max_size
        p = self.TestPool(**kwargs)
        if self.create_error:

            def create_error(retry=None):
                raise RuntimeError
            orig_create = p.create
            self.useFixture(fixtures.MockPatchObject(p, 'create', create_error))
            self.assertRaises(RuntimeError, p.get)
            self.useFixture(fixtures.MockPatchObject(p, 'create', orig_create))
        objs = []
        for i in range(self.n_iters):
            objs.append(p.get())
            self.assertIsInstance(objs[i], uuid.UUID)

        def wait_for_obj():
            o = p.get()
            self.assertIn(o, objs)
        waiter = self.ThreadWaitWaiter(p._cond, self.n_iters, self)
        threads = []
        for i in range(self.n_iters):
            t = threading.Thread(target=wait_for_obj)
            t.start()
            threads.append(t)
        waiter.wait()
        for o in objs:
            p.put(o)
        for t in threads:
            t.join()
        for o in objs:
            p.put(o)
        for o in p.iter_free():
            self.assertIn(o, objs)
            objs.remove(o)
        self.assertEqual([], objs)