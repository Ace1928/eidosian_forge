from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
@property
def unused_data(self):
    """Bytes past the end of compressed data.

        If ``decompress()`` is fed additional data beyond the end of a zstd
        frame, this value will be non-empty once ``decompress()`` fully decodes
        the input frame.
        """
    return self._unused_input