import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
class DAVServer(http_server.HttpServer):
    """Subclass of HttpServer that gives http+webdav urls.

    This is for use in testing: connections to this server will always go
    through _urllib where possible.
    """

    def __init__(self):
        super().__init__(TestingDAVRequestHandler)
    _url_protocol = 'http+webdav'