import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def _get_threading_ident():
    if sys.version_info >= (3, 3):
        return threading.get_ident()
    return threading._get_ident()