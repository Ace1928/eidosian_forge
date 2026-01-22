import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
def _set_screen_handler(self, log, enable, stream=None):
    h = self._get_builtin_handler(log, 'screen')
    if enable:
        if not h:
            if stream is None:
                stream = sys.stderr
            h = logging.StreamHandler(stream)
            h.setFormatter(logfmt)
            h._cpbuiltin = 'screen'
            log.addHandler(h)
    elif h:
        log.handlers.remove(h)