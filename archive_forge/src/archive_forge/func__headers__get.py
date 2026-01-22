import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def _headers__get(self):
    """
        All the request headers as a case-insensitive dictionary-like
        object.
        """
    if self._headers is None:
        self._headers = EnvironHeaders(self.environ)
    return self._headers