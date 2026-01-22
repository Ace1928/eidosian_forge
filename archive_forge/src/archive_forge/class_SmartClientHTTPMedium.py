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
class SmartClientHTTPMedium(medium.SmartClientMedium):

    def __init__(self, http_transport):
        super().__init__(http_transport.base)
        self._http_transport_ref = weakref.ref(http_transport)

    def get_request(self):
        return SmartClientHTTPMediumRequest(self)

    def should_probe(self):
        return True

    def remote_path_from_transport(self, transport):
        transport_base = transport.base
        if transport_base.startswith('bzr+'):
            transport_base = transport_base[4:]
        rel_url = urlutils.relative_url(self.base, transport_base)
        return urlutils.unquote(rel_url)

    def send_http_smart_request(self, bytes):
        try:
            t = self._http_transport_ref()
            code, body_filelike = t._post(bytes)
            if code != 200:
                raise errors.UnexpectedHttpStatus(t._remote_path('.bzr/smart'), code)
        except (errors.InvalidHttpResponse, errors.ConnectionReset) as e:
            raise errors.SmartProtocolError(str(e))
        return body_filelike

    def _report_activity(self, bytes, direction):
        """See SmartMedium._report_activity.

        Does nothing; the underlying plain HTTP transport will report the
        activity that this medium would report.
        """
        pass

    def disconnect(self):
        """See SmartClientMedium.disconnect()."""
        t = self._http_transport_ref()
        t.disconnect()