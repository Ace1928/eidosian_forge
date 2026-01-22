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
def _redirected_to(self, source, target):
    """Returns a transport suitable to re-issue a redirected request.

        :param source: The source url as returned by the server.
        :param target: The target url as returned by the server.

        The redirection can be handled only if the relpath involved is not
        renamed by the redirection.

        :returns: A transport
        :raise UnusableRedirect: when the URL can not be reinterpreted
        """
    parsed_source = self._split_url(source)
    parsed_target = self._split_url(target)
    pl = len(self._parsed_url.path)
    excess_tail = parsed_source.path[pl:].strip('/')
    if not parsed_target.path.endswith(excess_tail):
        raise UnusableRedirect(source, target, 'final part of the url was renamed')
    target_path = parsed_target.path
    if excess_tail:
        target_path = target_path[:-len(excess_tail)]
    if parsed_target.scheme in ('http', 'https'):
        if parsed_target.scheme == self._unqualified_scheme and parsed_target.host == self._parsed_url.host and (parsed_target.port == self._parsed_url.port) and (parsed_target.user is None or parsed_target.user == self._parsed_url.user):
            return self.clone(target_path)
        else:
            redir_scheme = parsed_target.scheme
            new_url = self._unsplit_url(redir_scheme, self._parsed_url.user, self._parsed_url.password, parsed_target.host, parsed_target.port, target_path)
            return transport.get_transport_from_url(new_url)
    else:
        new_url = self._unsplit_url(parsed_target.scheme, parsed_target.user, parsed_target.password, parsed_target.host, parsed_target.port, target_path)
        return transport.get_transport_from_url(new_url)