from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def unicode_data_no_special_types(self):
    """Target database/dialect can receive / deliver / compare data with
        non-ASCII characters in plain VARCHAR, TEXT columns, without the need
        for special "national" datatypes like NVARCHAR or similar.

        """
    return exclusions.open()