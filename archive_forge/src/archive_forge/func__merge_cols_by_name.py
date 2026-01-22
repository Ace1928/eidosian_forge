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
def _merge_cols_by_name(self, context, cursor_description, result_columns, loose_column_name_matching):
    match_map = self._create_description_match_map(result_columns, loose_column_name_matching)
    mapped_type: TypeEngine[Any]
    for idx, colname, untranslated, coltype in self._colnames_from_description(context, cursor_description):
        try:
            ctx_rec = match_map[colname]
        except KeyError:
            mapped_type = sqltypes.NULLTYPE
            obj = None
            result_columns_idx = None
        else:
            obj = ctx_rec[1]
            mapped_type = ctx_rec[2]
            result_columns_idx = ctx_rec[3]
        yield (idx, result_columns_idx, colname, mapped_type, coltype, obj, untranslated)