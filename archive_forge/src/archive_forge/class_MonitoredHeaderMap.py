import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
class MonitoredHeaderMap(_httputil.HeaderMap):

    def transform_key(self, key):
        self.accessed_headers.add(key)
        return super(MonitoredHeaderMap, self).transform_key(key)

    def __init__(self):
        self.accessed_headers = set()
        super(MonitoredHeaderMap, self).__init__()