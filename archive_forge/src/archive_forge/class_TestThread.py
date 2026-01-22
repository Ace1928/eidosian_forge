import threading
from .. import cethread, tests
class TestThread(cethread.CatchingExceptionThread):

    def __init__(self, *args, **kwargs):
        self.step1 = threading.Event()
        self.step2 = threading.Event()
        super().__init__(target=self.step_by_step, sync_event=self.step1)
        self.current_step = 'starting'
        self.set_sync_event(self.step1)

    def step_by_step(self):
        control1.wait()
        self.current_step = 'step1'
        self.switch_and_set(self.step2)

    def set_sync_event(self, event):
        if event is self.step2:
            raise MyException()
        super().set_sync_event(event)