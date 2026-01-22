from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
def _adapt_update_db_opts(fn):
    insp = util.inspect_getfullargspec(fn)
    if len(insp.args) == 3:
        return fn
    else:
        return lambda db_url, db_opts, _options: fn(db_url, db_opts)