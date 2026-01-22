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
class ReconnectFixture:

    def __init__(self, dbapi):
        self.dbapi = dbapi
        self.connections = []
        self.is_stopped = False

    def __getattr__(self, key):
        return getattr(self.dbapi, key)

    def connect(self, *args, **kwargs):
        conn = self.dbapi.connect(*args, **kwargs)
        if self.is_stopped:
            self._safe(conn.close)
            curs = conn.cursor()
            curs.execute('select 1')
            assert False, "simulated connect failure didn't work"
        else:
            self.connections.append(conn)
            return conn

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            warnings.warn("ReconnectFixture couldn't close connection: %s" % e)

    def shutdown(self, stop=False):
        self.is_stopped = stop
        for c in list(self.connections):
            self._safe(c.close)
        self.connections = []

    def restart(self):
        self.is_stopped = False