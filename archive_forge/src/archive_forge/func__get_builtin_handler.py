import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
def _get_builtin_handler(self, log, key):
    for h in log.handlers:
        if getattr(h, '_cpbuiltin', None) == key:
            return h