from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
def _get_reflection_info(self, schema: Optional[str]=None, filter_names: Optional[Collection[str]]=None, available: Optional[Collection[str]]=None, _reflect_info: Optional[_ReflectionInfo]=None, **kw: Any) -> _ReflectionInfo:
    kw['schema'] = schema
    if filter_names and available and (len(filter_names) > 100):
        fraction = len(filter_names) / len(available)
    else:
        fraction = None
    unreflectable: Dict[TableKey, exc.UnreflectableTableError]
    kw['unreflectable'] = unreflectable = {}
    has_result: bool = True

    def run(meth: Any, *, optional: bool=False, check_filter_names_from_meth: bool=False) -> Any:
        nonlocal has_result
        if fraction is None or fraction <= 0.5 or (not self.dialect._overrides_default(meth.__name__)):
            _fn = filter_names
        else:
            _fn = None
        try:
            if has_result:
                res = meth(filter_names=_fn, **kw)
                if check_filter_names_from_meth and (not res):
                    has_result = False
            else:
                res = {}
        except NotImplementedError:
            if not optional:
                raise
            res = {}
        return res
    info = _ReflectionInfo(columns=run(self.get_multi_columns, check_filter_names_from_meth=True), pk_constraint=run(self.get_multi_pk_constraint), foreign_keys=run(self.get_multi_foreign_keys), indexes=run(self.get_multi_indexes), unique_constraints=run(self.get_multi_unique_constraints, optional=True), table_comment=run(self.get_multi_table_comment, optional=True), check_constraints=run(self.get_multi_check_constraints, optional=True), table_options=run(self.get_multi_table_options, optional=True), unreflectable=unreflectable)
    if _reflect_info:
        _reflect_info.update(info)
        return _reflect_info
    else:
        return info