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
def _urlvars__set(self, value):
    environ = self.environ
    if 'wsgiorg.routing_args' in environ:
        environ['wsgiorg.routing_args'] = (environ['wsgiorg.routing_args'][0], value)
        if 'paste.urlvars' in environ:
            del environ['paste.urlvars']
    elif 'paste.urlvars' in environ:
        environ['paste.urlvars'] = value
    else:
        environ['wsgiorg.routing_args'] = ((), value)