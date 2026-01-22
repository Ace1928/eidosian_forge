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
def _urlvars__del(self):
    if 'paste.urlvars' in self.environ:
        del self.environ['paste.urlvars']
    if 'wsgiorg.routing_args' in self.environ:
        if not self.environ['wsgiorg.routing_args'][0]:
            del self.environ['wsgiorg.routing_args']
        else:
            self.environ['wsgiorg.routing_args'] = (self.environ['wsgiorg.routing_args'][0], {})