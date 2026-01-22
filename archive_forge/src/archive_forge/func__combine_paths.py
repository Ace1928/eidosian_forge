import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
@staticmethod
def _combine_paths(base_path: str, relpath: str) -> str:
    """Transform a Transport-relative path to a remote absolute path.

        This does not handle substitution of ~ but does handle '..' and '.'
        components.

        Examples::

            t._combine_paths('/home/sarah', 'project/foo')
                => '/home/sarah/project/foo'
            t._combine_paths('/home/sarah', '../../etc')
                => '/etc'
            t._combine_paths('/home/sarah', '/etc')
                => '/etc'

        Args:
          base_path: base path
          relpath: relative url string for relative part of remote path.
        Returns: urlencoded string for final path.
        """
    if not isinstance(relpath, str):
        raise InvalidURL(relpath)
    relpath = _url_hex_escapes_re.sub(_unescape_safe_chars, relpath)
    if relpath.startswith('/'):
        base_parts = []
    else:
        base_parts = base_path.split('/')
    if len(base_parts) > 0 and base_parts[-1] == '':
        base_parts = base_parts[:-1]
    for p in relpath.split('/'):
        if p == '..':
            if len(base_parts) == 0:
                continue
            base_parts.pop()
        elif p == '.':
            continue
        elif p != '':
            base_parts.append(p)
    path = '/'.join(base_parts)
    if not path.startswith('/'):
        path = '/' + path
    return path