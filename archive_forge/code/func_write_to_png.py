import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def write_to_png(self, target=None):
    """Writes the contents of surface as a PNG image.

        :param target:
            A filename,
            a binary mode :term:`file object` with a `write` method,
            or :obj:`None`.
        :returns:
            If ``target`` is :obj:`None`,
            return the PNG contents as a byte string.

        """
    return_bytes = target is None
    if return_bytes:
        target = io.BytesIO()
    if hasattr(target, 'write'):
        try:
            write_func = _make_write_func(target)
            _check_status(cairo.cairo_surface_write_to_png_stream(self._pointer, write_func, ffi.NULL))
        except (SystemError, MemoryError):
            if hasattr(target, 'name'):
                _check_status(cairo.cairo_surface_write_to_png(self._pointer, _encode_filename(target.name)))
            else:
                with NamedTemporaryFile('wb', delete=False) as fd:
                    filename = fd.name
                    _check_status(cairo.cairo_surface_write_to_png(self._pointer, _encode_filename(filename)))
                png_file = Path(filename)
                target.write(png_file.read_bytes())
                png_file.unlink()
    else:
        _check_status(cairo.cairo_surface_write_to_png(self._pointer, _encode_filename(target)))
    if return_bytes:
        return target.getvalue()