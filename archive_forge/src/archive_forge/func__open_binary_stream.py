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
def _open_binary_stream(uri, mode, transport_params):
    """Open an arbitrary URI in the specified binary mode.

    Not all modes are supported for all protocols.

    :arg uri: The URI to open.  May be a string, or something else.
    :arg str mode: The mode to open with.  Must be rb, wb or ab.
    :arg transport_params: Keyword argumens for the transport layer.
    :returns: A named file object
    :rtype: file-like object with a .name attribute
    """
    if mode not in ('rb', 'rb+', 'wb', 'wb+', 'ab', 'ab+'):
        raise NotImplementedError('unsupported mode: %r' % mode)
    if isinstance(uri, int):
        fobj = _builtin_open(uri, mode, closefd=False)
        return fobj
    if not isinstance(uri, str):
        raise TypeError("don't know how to handle uri %s" % repr(uri))
    scheme = _sniff_scheme(uri)
    submodule = transport.get_transport(scheme)
    fobj = submodule.open_uri(uri, mode, transport_params)
    if not hasattr(fobj, 'name'):
        fobj.name = uri
    return fobj