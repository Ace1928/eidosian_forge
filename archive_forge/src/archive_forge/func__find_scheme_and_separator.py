import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _find_scheme_and_separator(url):
    """Find the scheme separator (://) and the first path separator

    This is just a helper functions for other path utilities.
    It could probably be replaced by urlparse
    """
    m = _url_scheme_re.match(url)
    if not m:
        return (None, None)
    scheme = m.group('scheme')
    path = m.group('path')
    first_path_slash = path.find('/')
    if first_path_slash == -1:
        return (len(scheme), None)
    return (len(scheme), first_path_slash + m.start('path'))