import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
@classmethod
def _file_for_file_module(cls, module):
    """Return the file for the module."""
    try:
        return module.__file__ and cls._make_absolute(module.__file__)
    except AttributeError:
        pass