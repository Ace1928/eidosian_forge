from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from . import _core
from ._abc import ReceiveStream, SendStream
from ._core._windows_cffi import _handle, kernel32, raise_winerror
from ._util import ConflictDetector, final
Represents a receive stream over an os.pipe object.