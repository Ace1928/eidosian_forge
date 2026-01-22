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
class _NoResultMetaData(ResultMetaData):
    __slots__ = ()
    returns_rows = False

    def _we_dont_return_rows(self, err=None):
        raise exc.ResourceClosedError('This result object does not return rows. It has been closed automatically.') from err

    def _index_for_key(self, keys, raiseerr):
        self._we_dont_return_rows()

    def _metadata_for_keys(self, key):
        self._we_dont_return_rows()

    def _reduce(self, keys):
        self._we_dont_return_rows()

    @property
    def _keymap(self):
        self._we_dont_return_rows()

    @property
    def _key_to_index(self):
        self._we_dont_return_rows()

    @property
    def _processors(self):
        self._we_dont_return_rows()

    @property
    def keys(self):
        self._we_dont_return_rows()