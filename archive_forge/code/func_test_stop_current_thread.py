import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_stop_current_thread(self):
    stop_event = event.Event()
    quit_event = event.Event()

    def stop_self(*args, **kwargs):
        if args[0] == 1:
            time.sleep(1)
            self.tg.stop()
            stop_event.send('stop_event')
        quit_event.wait()
    for i in range(0, 4):
        self.tg.add_thread(stop_self, i, kwargs='kwargs')
    stop_event.wait()
    self.assertEqual(1, len(self.tg.threads))
    quit_event.send('quit_event')