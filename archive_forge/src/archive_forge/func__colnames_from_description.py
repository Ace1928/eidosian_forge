from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
def _colnames_from_description(self, context, cursor_description):
    """Extract column names and data types from a cursor.description.

        Applies unicode decoding, column translation, "normalization",
        and case sensitivity rules to the names based on the dialect.

        """
    dialect = context.dialect
    translate_colname = context._translate_colname
    normalize_name = dialect.normalize_name if dialect.requires_name_normalize else None
    untranslated = None
    self._keys = []
    for idx, rec in enumerate(cursor_description):
        colname = rec[0]
        coltype = rec[1]
        if translate_colname:
            colname, untranslated = translate_colname(colname)
        if normalize_name:
            colname = normalize_name(colname)
        self._keys.append(colname)
        yield (idx, colname, untranslated, coltype)