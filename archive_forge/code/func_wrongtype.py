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
@cherrypy.expose
@cherrypy.config(**{'tools.expires.secs': Foo()})
def wrongtype(self):
    cherrypy.response.headers['Etag'] = 'need_this_to_make_me_cacheable'
    return 'Woops'