import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def _require_signal_and_kill(self, signal_name):
    if not hasattr(signal, signal_name):
        self.skip('skipped (no %(signal_name)s)' % vars())
    if not hasattr(os, 'kill'):
        self.skip('skipped (no os.kill)')