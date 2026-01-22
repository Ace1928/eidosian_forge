import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
@cherrypy.config(**{'tools.caching.on': True, 'tools.response_headers.on': True, 'tools.response_headers.headers': [('Vary', 'Our-Varying-Header')]})
class VaryHeaderCachingServer(object):

    def __init__(self):
        self.counter = count(1)

    @cherrypy.expose
    def index(self):
        return 'visit #%s' % next(self.counter)