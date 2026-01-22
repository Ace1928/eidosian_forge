from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
def _running_on_windows(self):
    return exclusions.LambdaPredicate(lambda: platform.system() == 'Windows', description='running on Windows')