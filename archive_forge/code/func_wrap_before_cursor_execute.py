from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from .base import Connection
from .base import Engine
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPIConnection
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .. import event
from .. import exc
from ..util.typing import Literal
def wrap_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    orig_fn(conn, cursor, statement, parameters, context, executemany)
    return (statement, parameters)