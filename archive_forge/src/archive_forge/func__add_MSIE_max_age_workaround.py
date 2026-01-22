import sys
import datetime
import os
import time
import threading
import binascii
import pickle
import zc.lockfile
import cherrypy
from cherrypy.lib import httputil
from cherrypy.lib import locking
from cherrypy.lib import is_iterator
def _add_MSIE_max_age_workaround(cookie, timeout):
    """
    We'd like to use the "max-age" param as indicated in
    http://www.faqs.org/rfcs/rfc2109.html but IE doesn't
    save it to disk and the session is lost if people close
    the browser. So we have to use the old "expires" ... sigh ...
    """
    expires = time.time() + timeout * 60
    cookie['expires'] = httputil.HTTPDate(expires)