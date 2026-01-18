import threading
from .. import cethread, tests
def test_switch_and_set(self):
    """Caller can precisely control a thread."""
    control1 = threading.Event()
    control2 = threading.Event()
    control3 = threading.Event()

    class TestThread(cethread.CatchingExceptionThread):

        def __init__(self):
            super().__init__(target=self.step_by_step)
            self.current_step = 'starting'
            self.step1 = threading.Event()
            self.set_sync_event(self.step1)
            self.step2 = threading.Event()
            self.final = threading.Event()

        def step_by_step(self):
            control1.wait()
            self.current_step = 'step1'
            self.switch_and_set(self.step2)
            control2.wait()
            self.current_step = 'step2'
            self.switch_and_set(self.final)
            control3.wait()
            self.current_step = 'done'
    tt = TestThread()
    tt.start()
    self.assertEqual('starting', tt.current_step)
    control1.set()
    tt.step1.wait()
    self.assertEqual('step1', tt.current_step)
    control2.set()
    tt.step2.wait()
    self.assertEqual('step2', tt.current_step)
    control3.set()
    tt.join()
    self.assertEqual('done', tt.current_step)