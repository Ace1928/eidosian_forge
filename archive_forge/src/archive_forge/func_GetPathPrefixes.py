from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
def GetPathPrefixes(path):
    """Returns all prefixes for the given path, inclusive.

  That is, for 'foo/bar/baz', returns ['', 'foo', 'foo/bar', 'foo/bar/baz'].

  Args:
    path: str, the path for which to get prefixes.

  Returns:
    list of str, the prefixes.
  """
    path_prefixes = [path]
    path_reminder = True
    while path and path_reminder:
        path, path_reminder = os.path.split(path)
        path_prefixes.insert(0, path)
    return path_prefixes