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
def a_gif(self):
    cherrypy.response.headers['Last-Modified'] = httputil.HTTPDate()
    return gif_bytes