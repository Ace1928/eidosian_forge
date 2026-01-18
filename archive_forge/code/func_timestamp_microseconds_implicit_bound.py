from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def timestamp_microseconds_implicit_bound(self):
    """target dialect when given a datetime object which also includes
        a microseconds portion when using the TIMESTAMP data type
        will bind it such that the database server knows
        the object is a datetime with microseconds, and not a plain string.

        """
    return self.timestamp_microseconds