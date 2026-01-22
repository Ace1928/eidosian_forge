from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
class _SerializableColumnGetterV2(_PlainColumnGetter[_KT]):
    """Updated serializable getter which deals with
    multi-table mapped classes.

    Two extremely unusual cases are not supported.
    Mappings which have tables across multiple metadata
    objects, or which are mapped to non-Table selectables
    linked across inheriting mappers may fail to function
    here.

    """
    __slots__ = ('colkeys',)

    def __init__(self, colkeys: Sequence[Tuple[Optional[str], Optional[str]]]) -> None:
        self.colkeys = colkeys
        self.composite = len(colkeys) > 1

    def __reduce__(self) -> Tuple[Type[_SerializableColumnGetterV2[_KT]], Tuple[Sequence[Tuple[Optional[str], Optional[str]]]]]:
        return (self.__class__, (self.colkeys,))

    @classmethod
    def _reduce_from_cols(cls, cols: Sequence[ColumnElement[_KT]]) -> Tuple[Type[_SerializableColumnGetterV2[_KT]], Tuple[Sequence[Tuple[Optional[str], Optional[str]]]]]:

        def _table_key(c: ColumnElement[_KT]) -> Optional[str]:
            if not isinstance(c.table, expression.TableClause):
                return None
            else:
                return c.table.key
        colkeys = [(c.key, _table_key(c)) for c in cols]
        return (_SerializableColumnGetterV2, (colkeys,))

    def _cols(self, mapper: Mapper[_KT]) -> Sequence[ColumnElement[_KT]]:
        cols: List[ColumnElement[_KT]] = []
        metadata = getattr(mapper.local_table, 'metadata', None)
        for ckey, tkey in self.colkeys:
            if tkey is None or metadata is None or tkey not in metadata:
                cols.append(mapper.local_table.c[ckey])
            else:
                cols.append(metadata.tables[tkey].c[ckey])
        return cols