import itertools
import platform
import threading
from http.client import HTTPConnection
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.test import helper
Tests for refleaks.