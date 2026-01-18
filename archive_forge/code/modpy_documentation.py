import os
import re
import cherrypy
from cherrypy.test import helper
Gracefully shutdown a server that is serving forever.