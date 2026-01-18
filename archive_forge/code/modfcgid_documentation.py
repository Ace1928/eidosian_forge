import os
import re
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.process import servers
from cherrypy.test import helper
Gracefully shutdown a server that is serving forever.