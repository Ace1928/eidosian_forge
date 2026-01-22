from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def _encode_string(string):
    """Return a byte string, encoding Unicode with UTF-8."""
    if not isinstance(string, bytes):
        string = string.encode('utf8')
    return ffi.new('char[]', string)