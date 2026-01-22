import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class Unfinished(RequestFromCodeRunner):
    """Source code wasn't executed because it wasn't fully formed"""