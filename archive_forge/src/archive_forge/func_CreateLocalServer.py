from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import select
import socket
import sys
import webbrowser
import wsgiref
from google_auth_oauthlib import flow as google_auth_flow
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as c_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import pkg_resources
from oauthlib.oauth2.rfc6749 import errors as rfc6749_errors
from requests import exceptions as requests_exceptions
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves.urllib import parse
def CreateLocalServer(wsgi_app, host, search_start_port, search_end_port):
    """Creates a local wsgi server.

  Finds an available port in the range of [search_start_port, search_end_point)
  for the local server.

  Args:
    wsgi_app: A wsgi app running on the local server.
    host: hostname of the server.
    search_start_port: int, the port where the search starts.
    search_end_port: int, the port where the search ends.

  Raises:
    LocalServerCreationError: If it cannot find an available port for
      the local server.

  Returns:
    WSGISever, a wsgi server.
  """
    port = search_start_port
    local_server = None
    while not local_server and port < search_end_port:
        try:
            local_server = wsgiref.simple_server.make_server(host, port, wsgi_app, server_class=WSGIServer, handler_class=google_auth_flow._WSGIRequestHandler)
        except (socket.error, OSError):
            port += 1
    if local_server:
        return local_server
    raise LocalServerCreationError(_PORT_SEARCH_ERROR_MSG.format(start_port=search_start_port, end_port=search_end_port - 1))