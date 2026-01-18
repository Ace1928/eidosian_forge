from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def unicode_ddl(self):
    """Target driver must support some degree of non-ascii symbol
        names.
        """
    return exclusions.closed()