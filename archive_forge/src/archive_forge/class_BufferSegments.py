from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class BufferSegments:
    """Represents an array of ``(offset, length)`` integers.

    This type is effectively an index used by :py:class:`BufferWithSegments`.

    The array members are 64-bit unsigned integers using host/native bit order.

    Instances conform to the buffer protocol.
    """