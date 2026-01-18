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
@register.init_decorator(_adapt_update_db_opts)
def update_db_opts(db_url, db_opts, options):
    """Set database options (db_opts) for a test database that we created."""