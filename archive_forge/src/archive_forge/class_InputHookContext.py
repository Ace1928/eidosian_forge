from __future__ import unicode_literals
import os
import threading
from prompt_toolkit.utils import is_windows
from .select import select_fds
class InputHookContext(object):
    """
    Given as a parameter to the inputhook.
    """

    def __init__(self, inputhook):
        assert callable(inputhook)
        self.inputhook = inputhook
        self._input_is_ready = None
        self._r, self._w = os.pipe()

    def input_is_ready(self):
        """
        Return True when the input is ready.
        """
        return self._input_is_ready(wait=False)

    def fileno(self):
        """
        File descriptor that will become ready when the event loop needs to go on.
        """
        return self._r

    def call_inputhook(self, input_is_ready_func):
        """
        Call the inputhook. (Called by a prompt-toolkit eventloop.)
        """
        self._input_is_ready = input_is_ready_func

        def thread():
            input_is_ready_func(wait=True)
            os.write(self._w, b'x')
        threading.Thread(target=thread).start()
        self.inputhook(self)
        try:
            if not is_windows():
                select_fds([self._r], timeout=None)
            os.read(self._r, 1024)
        except OSError:
            pass
        self._input_is_ready = None

    def close(self):
        """
        Clean up resources.
        """
        if self._r:
            os.close(self._r)
            os.close(self._w)
        self._r = self._w = None