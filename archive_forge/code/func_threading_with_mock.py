from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def threading_with_mock(self):
    """Mark tests that use threading and mock at the same time - stability
        issues have been observed with coverage

        """
    return exclusions.skip_if(lambda config: config.options.has_coverage, 'Stability issues with coverage')