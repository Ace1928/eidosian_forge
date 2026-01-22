import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
def _set_wsgi_handler(self, log, enable):
    h = self._get_builtin_handler(log, 'wsgi')
    if enable:
        if not h:
            h = WSGIErrorHandler()
            h.setFormatter(logfmt)
            h._cpbuiltin = 'wsgi'
            log.addHandler(h)
    elif h:
        log.handlers.remove(h)