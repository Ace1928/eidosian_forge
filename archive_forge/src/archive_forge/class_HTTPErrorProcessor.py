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
class HTTPErrorProcessor(urllib.request.HTTPErrorProcessor):
    """Process HTTP error responses.

    We don't really process the errors, quite the contrary
    instead, we leave our Transport handle them.
    """
    accepted_errors = [200, 201, 202, 204, 206, 207, 400, 403, 404, 405, 406, 409, 412, 416, 422, 501]
    'The error codes the caller will handle.\n\n    This can be specialized in the request on a case-by case basis, but the\n    common cases are covered here.\n    '

    def http_response(self, request, response):
        code, msg, hdrs = (response.code, response.msg, response.info())
        if code not in self.accepted_errors:
            response = self.parent.error('http', request, response, code, msg, hdrs)
        return response
    https_response = http_response