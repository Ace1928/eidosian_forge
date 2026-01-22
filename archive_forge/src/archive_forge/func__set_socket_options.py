from __future__ import absolute_import
import socket
from urllib3.contrib import _appengine_environ
from urllib3.exceptions import LocationParseError
from urllib3.packages import six
from .wait import NoWayToWaitForSocketError, wait_for_read
def _set_socket_options(sock, options):
    if options is None:
        return
    for opt in options:
        sock.setsockopt(*opt)