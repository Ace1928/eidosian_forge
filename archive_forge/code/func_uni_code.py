import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
@cherrypy.expose
def uni_code(self):
    cherrypy.request.login = tartaros
    cherrypy.request.remote.name = erebos