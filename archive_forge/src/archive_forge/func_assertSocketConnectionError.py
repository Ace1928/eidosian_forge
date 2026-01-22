import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def assertSocketConnectionError(self, expected, *args, **kwargs):
    """Check the formatting of a SocketConnectionError exception"""
    e = errors.SocketConnectionError(*args, **kwargs)
    self.assertEqual(expected, str(e))