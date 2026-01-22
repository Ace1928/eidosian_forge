import collections
import io
import locale
import logging
import os
import os.path as P
import pathlib
import urllib.parse
import warnings
import smart_open.local_file as so_file
import smart_open.compression as so_compression
from smart_open import doctools
from smart_open import transport
from smart_open.compression import register_compressor  # noqa: F401
from smart_open.utils import check_kwargs as _check_kwargs  # noqa: F401
from smart_open.utils import inspect_kwargs as _inspect_kwargs  # noqa: F401
def _shortcut_open(uri, mode, compression, buffering=-1, encoding=None, errors=None, newline=None):
    """Try to open the URI using the standard library io.open function.

    This can be much faster than the alternative of opening in binary mode and
    then decoding.

    This is only possible under the following conditions:

        1. Opening a local file; and
        2. Compression is disabled

    If it is not possible to use the built-in open for the specified URI, returns None.

    :param str uri: A string indicating what to open.
    :param str mode: The mode to pass to the open function.
    :param str compression: The compression type selected.
    :returns: The opened file
    :rtype: file
    """
    if not isinstance(uri, str):
        return None
    scheme = _sniff_scheme(uri)
    if scheme not in (transport.NO_SCHEME, so_file.SCHEME):
        return None
    local_path = so_file.extract_local_path(uri)
    if compression == so_compression.INFER_FROM_EXTENSION:
        _, extension = P.splitext(local_path)
        if extension in so_compression.get_supported_extensions():
            return None
    elif compression != so_compression.NO_COMPRESSION:
        return None
    open_kwargs = {}
    if encoding is not None:
        open_kwargs['encoding'] = encoding
        mode = mode.replace('b', '')
    if newline is not None:
        open_kwargs['newline'] = newline
    if errors and 'b' not in mode:
        open_kwargs['errors'] = errors
    return _builtin_open(local_path, mode, buffering=buffering, **open_kwargs)