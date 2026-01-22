from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
def _bulk_update(mapper: Mapper[Any], mappings: Union[Iterable[InstanceState[_O]], Iterable[Dict[str, Any]]], session_transaction: SessionTransaction, isstates: bool, update_changed_only: bool, use_orm_update_stmt: Optional[dml.Update]=None, enable_check_rowcount: bool=True) -> Optional[_result.Result[Any]]:
    base_mapper = mapper.base_mapper
    search_keys = mapper._primary_key_propkeys
    if mapper._version_id_prop:
        search_keys = {mapper._version_id_prop.key}.union(search_keys)

    def _changed_dict(mapper, state):
        return {k: v for k, v in state.dict.items() if k in state.committed_state or k in search_keys}
    if isstates:
        if update_changed_only:
            mappings = [_changed_dict(mapper, state) for state in mappings]
        else:
            mappings = [state.dict for state in mappings]
    else:
        mappings = [dict(m) for m in mappings]
        _expand_composites(mapper, mappings)
    if session_transaction.session.connection_callable:
        raise NotImplementedError('connection_callable / per-instance sharding not supported in bulk_update()')
    connection = session_transaction.connection(base_mapper)
    extra_bp_names = [b.key for b in use_orm_update_stmt._get_embedded_bindparams() if b.key in mappings[0]] if use_orm_update_stmt is not None else ()
    for table, super_mapper in base_mapper._sorted_tables.items():
        if not mapper.isa(super_mapper) or table not in mapper._pks_by_table:
            continue
        records = persistence._collect_update_commands(None, table, ((None, mapping, mapper, connection, mapping[mapper._version_id_prop.key] if mapper._version_id_prop else None) for mapping in mappings), bulk=True, use_orm_update_stmt=use_orm_update_stmt, include_bulk_keys=extra_bp_names)
        persistence._emit_update_statements(base_mapper, None, super_mapper, table, records, bookkeeping=False, use_orm_update_stmt=use_orm_update_stmt, enable_check_rowcount=enable_check_rowcount)
    if use_orm_update_stmt is not None:
        return _result.null_result()