import ctypes
import signal
import threading
class TimerDeathPenalty(BaseDeathPenalty):

    def __init__(self, timeout, exception=JobTimeoutException, **kwargs):
        super().__init__(timeout, exception, **kwargs)
        self._target_thread_id = threading.current_thread().ident
        self._timer = None

        def init_with_message(self, *args, **kwargs):
            super(exception, self).__init__('Task exceeded maximum timeout value ({0} seconds)'.format(timeout))
        self._exception.__init__ = init_with_message

    def new_timer(self):
        """Returns a new timer since timers can only be used once."""
        return threading.Timer(self._timeout, self.handle_death_penalty)

    def handle_death_penalty(self):
        """Raises an asynchronous exception in another thread.

        Reference http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc for more info.
        """
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._target_thread_id), ctypes.py_object(self._exception))
        if ret == 0:
            raise ValueError('Invalid thread ID {}'.format(self._target_thread_id))
        elif ret > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._target_thread_id), 0)
            raise SystemError('PyThreadState_SetAsyncExc failed')

    def setup_death_penalty(self):
        """Starts the timer."""
        if self._timeout <= 0:
            return
        self._timer = self.new_timer()
        self._timer.start()

    def cancel_death_penalty(self):
        """Cancels the timer."""
        if self._timeout <= 0:
            return
        self._timer.cancel()
        self._timer = None