from base64 import standard_b64encode
from distutils import log
from distutils.errors import DistutilsOptionError
import os
import zipfile
import tempfile
import shutil
import itertools
import functools
import http.client
import urllib.parse
from .._importlib import metadata
from ..warnings import SetuptoolsDeprecationWarning
from .upload import upload
@classmethod
def _build_multipart(cls, data):
    """
        Build up the MIME payload for the POST data
        """
    boundary = '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254'
    sep_boundary = b'\n--' + boundary.encode('ascii')
    end_boundary = sep_boundary + b'--'
    end_items = (end_boundary, b'\n')
    builder = functools.partial(cls._build_part, sep_boundary=sep_boundary)
    part_groups = map(builder, data.items())
    parts = itertools.chain.from_iterable(part_groups)
    body_items = itertools.chain(parts, end_items)
    content_type = 'multipart/form-data; boundary=%s' % boundary
    return (b''.join(body_items), content_type)