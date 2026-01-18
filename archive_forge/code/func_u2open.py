from suds.properties import Unskin
from suds.transport import *
import base64
from http.cookiejar import CookieJar
import http.client
import socket
import sys
import urllib.request, urllib.error, urllib.parse
import gzip
import zlib
from logging import getLogger
def u2open(self, u2request, timeout=None):
    """
        Open a connection.

        @param u2request: A urllib2 request.
        @type u2request: urllib2.Request.
        @return: The opened file-like urllib2 object.
        @rtype: fp

        """
    tm = timeout or self.options.timeout
    url = self.u2opener()
    return url.open(u2request, timeout=tm)