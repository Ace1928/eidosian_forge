import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def _head(self, relpath):
    """Request the HEAD of a file.

        Performs the request and leaves callers handle the results.
        """
    abspath = self._remote_path(relpath)
    response = self.request('HEAD', abspath)
    if response.status not in (200, 404):
        raise errors.UnexpectedHttpStatus(abspath, response.status, headers=response.getheaders())
    return response
    raise NotImplementedError(self._post)