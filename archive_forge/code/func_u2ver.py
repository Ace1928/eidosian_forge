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
def u2ver(self):
    """
        Get the major/minor version of the urllib2 lib.

        @return: The urllib2 version.
        @rtype: float

        """
    try:
        return float('%d.%d' % sys.version_info[:2])
    except Exception as e:
        return 0