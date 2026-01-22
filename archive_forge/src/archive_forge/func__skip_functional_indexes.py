from __future__ import annotations
import logging
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import cast
from sqlalchemy import schema
from sqlalchemy import text
from . import _autogen
from . import base
from ._autogen import _constraint_sig as _constraint_sig
from ._autogen import ComparisonResult as ComparisonResult
from .. import util
from ..util import sqla_compat
def _skip_functional_indexes(self, metadata_indexes, conn_indexes):
    conn_indexes_by_name = {c.name: c for c in conn_indexes}
    for idx in list(metadata_indexes):
        if idx.name in conn_indexes_by_name:
            continue
        iex = sqla_compat.is_expression_index(idx)
        if iex:
            util.warn(f"autogenerate skipping metadata-specified expression-based index {idx.name!r}; dialect {self.__dialect__!r} under SQLAlchemy {sqla_compat.sqlalchemy_version} can't reflect these indexes so they can't be compared")
            metadata_indexes.discard(idx)