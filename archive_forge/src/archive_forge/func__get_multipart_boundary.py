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
def _get_multipart_boundary(ctype):
    m = re.search('boundary=([^ ]+)', ctype, re.I)
    if m:
        return native_(m.group(1).strip('"'))