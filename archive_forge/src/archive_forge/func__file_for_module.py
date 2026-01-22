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
def _file_for_module(cls, module):
    """Return the relevant file for the module."""
    return cls._archive_for_zip_module(module) or cls._file_for_file_module(module)