from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def unusual_column_name_characters(self):
    """target database allows column names that have unusual characters
        in them, such as dots, spaces, slashes, or percent signs.

        The column names are as always in such a case quoted, however the
        DB still needs to support those characters in the name somehow.

        """
    return exclusions.open()