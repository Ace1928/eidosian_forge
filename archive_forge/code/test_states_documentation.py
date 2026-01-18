import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
SIGTERM should shut down the server whether daemonized or not.