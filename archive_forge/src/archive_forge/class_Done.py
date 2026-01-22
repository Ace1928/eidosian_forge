import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class Done(RequestFromCodeRunner):
    """Running code is done running"""