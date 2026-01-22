import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
class LazyRfc3339UtcTime(object):

    def __str__(self):
        """Return utcnow() in RFC3339 UTC Format."""
        iso_formatted_now = datetime.datetime.utcnow().isoformat('T')
        return f'{iso_formatted_now!s}Z'