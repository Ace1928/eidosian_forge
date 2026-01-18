import threading
from .. import cethread, tests
def test_sync_event(self):
    control = threading.Event()
    in_thread = threading.Event()

    class MyException(Exception):
        pass

    def raise_my_exception():
        control.wait()
        raise MyException()
    tt = cethread.CatchingExceptionThread(target=raise_my_exception, sync_event=in_thread)
    tt.start()
    tt.join(timeout=0)
    self.assertIs(None, tt.exception)
    self.assertIs(in_thread, tt.sync_event)
    control.set()
    self.assertRaises(MyException, tt.join)
    self.assertEqual(True, tt.sync_event.is_set())