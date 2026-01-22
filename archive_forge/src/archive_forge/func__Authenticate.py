import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
def _Authenticate(self):
    """Save the cookie jar after authentication."""
    if self.cert_file_available and (not can_validate_certs()):
        logger.warn('ssl module not found.\nWithout the ssl module, the identity of the remote host cannot be verified, and\nconnections may NOT be secure. To fix this, please install the ssl module from\nhttp://pypi.python.org/pypi/ssl .\nTo learn more, see https://developers.google.com/appengine/kb/general#rpcssl')
    super(HttpRpcServer, self)._Authenticate()
    if self.cookie_jar.filename is not None and self.save_cookies:
        logger.debug('Saving authentication cookies to %s', self.cookie_jar.filename)
        self.cookie_jar.save()
        self._CheckCookie()