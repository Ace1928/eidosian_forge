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
def _find_data_conn(self):
    """Finds the connection which will be used to send data."""
    sockets_to_read = [self.socket]
    client_connections = []
    while True:
        sockets_ready_to_read, _, _ = select.select(sockets_to_read, [], [])
        for s in sockets_ready_to_read:
            if s is self.socket:
                self._handle_new_client(s, sockets_to_read, client_connections)
            elif self._conn_closed(s):
                self._handle_closed_conn(s, sockets_to_read, client_connections)
            else:
                self._handle_non_data_conn(s, client_connections)
                return self._find_data_conn_with_client_address(s, client_connections)