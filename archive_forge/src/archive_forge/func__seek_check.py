from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def _seek_check(self, frame):
    if frame < self._min_frame or (not (hasattr(self, '_n_frames') and self._n_frames is None) and frame >= self.n_frames + self._min_frame):
        msg = 'attempt to seek outside sequence'
        raise EOFError(msg)
    return self.tell() != frame