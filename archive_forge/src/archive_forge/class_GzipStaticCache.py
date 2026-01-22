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
@cherrypy.config(**{'tools.gzip.mime_types': ['text/*', 'image/*'], 'tools.caching.on': True, 'tools.staticdir.on': True, 'tools.staticdir.dir': 'static', 'tools.staticdir.root': curdir})
class GzipStaticCache(object):
    pass