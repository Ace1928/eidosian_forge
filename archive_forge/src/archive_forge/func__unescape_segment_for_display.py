import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _unescape_segment_for_display(segment, encoding):
    """Unescape a segment for display.

    Helper for unescape_for_display

    Args:
      url: A 7-bit ASCII URL
      encoding: The final output encoding

    Returns: A unicode string which can be safely encoded into the
         specified encoding.
    """
    escaped_chunks = segment.split('%')
    escaped_chunks[0] = escaped_chunks[0].encode('utf-8')
    for j in range(1, len(escaped_chunks)):
        item = escaped_chunks[j]
        try:
            escaped_chunks[j] = _hex_display_map[item[:2]]
        except KeyError:
            escaped_chunks[j] = b'%' + item[:2].encode('utf-8')
        except UnicodeDecodeError:
            escaped_chunks[j] = chr(int(item[:2], 16)).encode('utf-8')
        escaped_chunks[j] += item[2:].encode('utf-8')
    unescaped = b''.join(escaped_chunks)
    try:
        decoded = unescaped.decode('utf-8')
    except UnicodeDecodeError:
        return segment
    else:
        try:
            decoded.encode(encoding)
        except UnicodeEncodeError:
            return segment
        else:
            return decoded