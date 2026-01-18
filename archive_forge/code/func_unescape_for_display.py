import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def unescape_for_display(url, encoding):
    """Decode what you can for a URL, so that we get a nice looking path.

    This will turn file:// urls into local paths, and try to decode
    any portions of a http:// style url that it can.

    Any sections of the URL which can't be represented in the encoding or
    need to stay as escapes are left alone.

    Args:
      url: A 7-bit ASCII URL
      encoding: The final output encoding

    Returns: A unicode string which can be safely encoded into the
         specified encoding.
    """
    if encoding is None:
        raise ValueError('you cannot specify None for the display encoding')
    if url.startswith('file://'):
        try:
            path = local_path_from_url(url)
            path.encode(encoding)
            return path
        except UnicodeError:
            return url
    res = url.split('/')
    for i in range(1, len(res)):
        res[i] = _unescape_segment_for_display(res[i], encoding)
    return '/'.join(res)