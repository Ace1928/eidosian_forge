from __future__ import absolute_import, print_function, division
import os
import io
import gzip
import sys
import bz2
import zipfile
from contextlib import contextmanager
import subprocess
import logging
from petl.errors import ArgumentError
from petl.compat import urlopen, StringIO, BytesIO, string_types, PY2
def _resolve_source_from_arg(source, handlers):
    if isinstance(source, string_types):
        handler = _get_handler_from(source, handlers)
        codec = _get_codec_for(source)
        if handler is None:
            if codec is not None:
                return codec(source)
            assert '://' not in source, _invalid_source_msg % source
            return FileSource(source)
        return handler(source)
    else:
        assert hasattr(source, 'open') and callable(getattr(source, 'open')), _invalid_source_msg % source
        return source