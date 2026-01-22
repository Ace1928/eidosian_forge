from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
def _drop_testing_engines(self, scope):
    eng = self.testing_engines[scope]
    for rec in list(eng):
        for proxy_ref in list(self.proxy_refs):
            if proxy_ref is not None and proxy_ref.is_valid:
                if proxy_ref._pool is not None and proxy_ref._pool is rec.pool:
                    self._safe(proxy_ref._checkin)
        if hasattr(rec, 'sync_engine'):
            await_only(rec.dispose())
        else:
            rec.dispose()
    eng.clear()