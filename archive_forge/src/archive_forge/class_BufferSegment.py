from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class BufferSegment:
    """Represents a segment within a ``BufferWithSegments``.

    This type is essentially a reference to N bytes within a
    ``BufferWithSegments``.

    The object conforms to the buffer protocol.
    """

    @property
    def offset(self):
        """The byte offset of this segment within its parent buffer."""
        raise NotImplementedError()

    def __len__(self):
        """Obtain the length of the segment, in bytes."""
        raise NotImplementedError()

    def tobytes(self):
        """Obtain bytes copy of this segment."""
        raise NotImplementedError()