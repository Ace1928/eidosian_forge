from .base import Clipboard, ClipboardData
from collections import deque

    Default clipboard implementation.
    Just keep the data in memory.

    This implements a kill-ring, for Emacs mode.
    