from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _decode_headers(headers, encoding):
    """
    Given an iterable of header two-tuples and an encoding, decodes those
    headers using that encoding while preserving the type of the header tuple.
    This ensures that the use of ``HeaderTuple`` is preserved.
    """
    for header in headers:
        assert isinstance(header, HeaderTuple)
        name, value = header
        name = name.decode(encoding)
        value = value.decode(encoding)
        yield header.__class__(name, value)