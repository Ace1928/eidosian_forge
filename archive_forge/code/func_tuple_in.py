from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def tuple_in(self):
    """Target platform supports the syntax
        "(x, y) IN ((x1, y1), (x2, y2), ...)"
        """
    return exclusions.closed()