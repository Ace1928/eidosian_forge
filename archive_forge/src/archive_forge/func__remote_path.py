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
def _remote_path(self, relpath):
    """See ConnectedTransport._remote_path.

        user and passwords are not embedded in the path provided to the server.
        """
    url = self._parsed_url.clone(relpath)
    url.user = url.quoted_user = None
    url.password = url.quoted_password = None
    url.scheme = self._unqualified_scheme
    return str(url)