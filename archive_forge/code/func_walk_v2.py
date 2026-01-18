import binascii
import os
from posixpath import join as urljoin
import uuid
import six
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('io.gfile.walk')
def walk_v2(top, topdown=True, onerror=None):
    """Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    topdown: bool, Traverse pre order if True, post order if False.
    onerror: optional handler for errors. Should be a function, it will be
      called with the error as argument. Rethrowing the error aborts the walk.
      Errors that happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files. That is, each yield looks like:
    `(dirname, [subdirname, subdirname, ...], [filename, filename, ...])`.
    Each item is a string.
  """

    def _make_full_path(parent, item):
        if item[0] == os.sep:
            return ''.join([join(parent, ''), item])
        return join(parent, item)
    top = compat.as_str_any(compat.path_to_str(top))
    try:
        listing = list_directory(top)
    except errors.NotFoundError as err:
        if onerror:
            onerror(err)
        else:
            return
    files = []
    subdirs = []
    for item in listing:
        full_path = _make_full_path(top, item)
        if is_directory(full_path):
            subdirs.append(item)
        else:
            files.append(item)
    here = (top, subdirs, files)
    if topdown:
        yield here
    for subdir in subdirs:
        for subitem in walk_v2(_make_full_path(top, subdir), topdown, onerror=onerror):
            yield subitem
    if not topdown:
        yield here