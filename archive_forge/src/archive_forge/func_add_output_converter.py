from __future__ import annotations
import collections
import itertools
from ..engine import AdaptedConnection
from ..util.concurrency import asyncio
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only
def add_output_converter(self, *arg, **kw):
    self._connection.add_output_converter(*arg, **kw)