import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
@staticmethod
def _archive_for_zip_module(module):
    """Return the archive filename for the module if relevant."""
    try:
        return module.__loader__.archive
    except AttributeError:
        pass