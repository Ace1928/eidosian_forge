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
def u2opener(self):
    """
        Create a urllib opener.

        @return: An opener.
        @rtype: I{OpenerDirector}

        """
    if self.urlopener is None:
        return urllib.request.build_opener(*self.u2handlers())
    return self.urlopener