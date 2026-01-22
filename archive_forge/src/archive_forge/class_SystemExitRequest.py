import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class SystemExitRequest(RequestFromCodeRunner):
    """Running code raised a SystemExit"""

    def __init__(self, args):
        self.args = args