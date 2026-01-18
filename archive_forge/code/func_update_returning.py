from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def update_returning(self):
    """target platform supports UPDATE ... RETURNING."""
    return exclusions.only_if(lambda config: config.db.dialect.update_returning, "%(database)s %(does_support)s 'UPDATE ... RETURNING'")