from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def append_path(self, path):
    """Append ``path`` onto the current path.
        The path may be either the return value from one of :meth:`copy_path`
        or :meth:`copy_path_flat` or it may be constructed manually.

        :param path:
            An iterable of tuples
            in the same format as returned by :meth:`copy_path`.

        """
    path, _ = _encode_path(path)
    cairo.cairo_append_path(self._pointer, path)
    self._check_status()