import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def _jython_SIGINT_handler(self, signum=None, frame=None):
    self.bus.log('Keyboard Interrupt: shutting down bus')
    self.bus.exit()