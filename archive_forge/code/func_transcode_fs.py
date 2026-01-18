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
def transcode_fs(self, fs, content_type):
    if PY2:

        def decode(b):
            if b is not None:
                return b.decode(self.charset, self.errors)
            else:
                return b
    else:

        def decode(b):
            return b
    data = []
    for field in fs.list or ():
        field.name = decode(field.name)
        if field.filename:
            field.filename = decode(field.filename)
            data.append((field.name, field))
        else:
            data.append((field.name, decode(field.value)))
    content_type, fout = _encode_multipart(data, content_type, fout=io.BytesIO())
    return fout