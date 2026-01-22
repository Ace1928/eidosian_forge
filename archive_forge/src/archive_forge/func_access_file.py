import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
@access_file.setter
def access_file(self, newvalue):
    self._set_file_handler(self.access_log, newvalue)