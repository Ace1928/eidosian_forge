import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
class QuirkyDAVServer(http_server.HttpServer):
    """DAVServer implementing various quirky/slightly off-spec behaviors.

    Used to test how gracefully we handle them.
    """

    def __init__(self):
        super().__init__(QuirkyTestingDAVRequestHandler)
    _url_protocol = 'http+webdav'