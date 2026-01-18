from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def unbounded_varchar(self):
    """Target database must support VARCHAR with no length"""
    return exclusions.open()