from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
class ImageOpener(Opener):
    """Opener-type class to collect extra compressed extensions

    A trivial sub-class of opener to which image classes can add extra
    extensions with custom openers, such as compressed openers.

    To add an extension, add a line to the class definition (not __init__):

        ImageOpener.compress_ext_map[ext] = func_def

    ``ext`` is a file extension beginning with '.' and should be included in
    the image class's ``valid_exts`` tuple.

    ``func_def`` is a `(function, (args,))` tuple, where `function accepts a
    filename as the first parameter, and `args` defines the other arguments
    that `function` accepts. These arguments must be any (unordered) subset of
    `mode`, `compresslevel`, and `buffering`.
    """
    compress_ext_map = Opener.compress_ext_map.copy()