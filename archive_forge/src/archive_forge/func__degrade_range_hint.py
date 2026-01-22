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
def _degrade_range_hint(self, relpath, ranges):
    if self._range_hint == 'multi':
        self._range_hint = 'single'
        mutter('Retry "%s" with single range request' % relpath)
    elif self._range_hint == 'single':
        self._range_hint = None
        mutter('Retry "%s" without ranges' % relpath)
    else:
        return False
    return True