import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_stop_gracefully(self):

    def foo(*args, **kwargs):
        time.sleep(1)
    start_time = time.time()
    self.tg.add_thread(foo, 'arg', kwarg='kwarg')
    self.tg.stop(True)
    end_time = time.time()
    self.assertEqual(0, len(self.tg.threads))
    self.assertTrue(end_time - start_time >= 1)
    self.assertEqual(0, len(self.tg.timers))