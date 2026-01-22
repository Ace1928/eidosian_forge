from __future__ import annotations
import functools
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import coercions
from . import dml
from . import elements
from . import roles
from .base import _DefaultDescriptionTuple
from .dml import isinsert as _compile_state_isinsert
from .elements import ColumnClause
from .schema import default_is_clause_element
from .schema import default_is_sequence
from .selectable import Select
from .selectable import TableClause
from .. import exc
from .. import util
from ..util.typing import Literal
def _get_crud_params(compiler: SQLCompiler, stmt: ValuesBase, compile_state: DMLState, toplevel: bool, **kw: Any) -> _CrudParams:
    """create a set of tuples representing column/string pairs for use
    in an INSERT or UPDATE statement.

    Also generates the Compiled object's postfetch, prefetch, and
    returning column collections, used for default handling and ultimately
    populating the CursorResult's prefetch_cols() and postfetch_cols()
    collections.

    """
    compiler.postfetch = []
    compiler.insert_prefetch = []
    compiler.update_prefetch = []
    compiler.implicit_returning = []
    visiting_cte = kw.get('visiting_cte', None)
    if visiting_cte is not None:
        kw.pop('accumulate_bind_names', None)
    assert 'accumulate_bind_names' not in kw, "Don't know how to handle insert within insert without a CTE"
    _column_as_key, _getattr_col_key, _col_bind_name = _key_getters_for_crud_column(compiler, stmt, compile_state)
    compiler._get_bind_name_for_col = _col_bind_name
    if stmt._returning and stmt._return_defaults:
        raise exc.CompileError("Can't compile statement that includes returning() and return_defaults() simultaneously")
    if compile_state.isdelete:
        _setup_delete_return_defaults(compiler, stmt, compile_state, (), _getattr_col_key, _column_as_key, _col_bind_name, (), (), toplevel, kw)
        return _CrudParams([], [])
    if compiler.column_keys is None and compile_state._no_parameters:
        return _CrudParams([(c, compiler.preparer.format_column(c), _create_bind_param(compiler, c, None, required=True), (c.key,)) for c in stmt.table.columns if not c._omit_from_statements], [])
    stmt_parameter_tuples: Optional[List[Tuple[Union[str, ColumnClause[Any]], Any]]]
    spd: Optional[MutableMapping[_DMLColumnElement, Any]]
    if _compile_state_isinsert(compile_state) and compile_state._has_multi_parameters:
        mp = compile_state._multi_parameters
        assert mp is not None
        spd = mp[0]
        stmt_parameter_tuples = list(spd.items())
        spd_str_key = {_column_as_key(key) for key in spd}
    elif compile_state._ordered_values:
        spd = compile_state._dict_parameters
        stmt_parameter_tuples = compile_state._ordered_values
        assert spd is not None
        spd_str_key = {_column_as_key(key) for key in spd}
    elif compile_state._dict_parameters:
        spd = compile_state._dict_parameters
        stmt_parameter_tuples = list(spd.items())
        spd_str_key = {_column_as_key(key) for key in spd}
    else:
        stmt_parameter_tuples = spd = spd_str_key = None
    if compiler.column_keys is None:
        parameters = {}
    elif stmt_parameter_tuples:
        assert spd_str_key is not None
        parameters = {_column_as_key(key): REQUIRED for key in compiler.column_keys if key not in spd_str_key}
    else:
        parameters = {_column_as_key(key): REQUIRED for key in compiler.column_keys}
    values: List[_CrudParamElement] = []
    if stmt_parameter_tuples is not None:
        _get_stmt_parameter_tuples_params(compiler, compile_state, parameters, stmt_parameter_tuples, _column_as_key, values, kw)
    check_columns: Dict[str, ColumnClause[Any]] = {}
    if dml.isupdate(compile_state) and compile_state.is_multitable:
        _get_update_multitable_params(compiler, stmt, compile_state, stmt_parameter_tuples, check_columns, _col_bind_name, _getattr_col_key, values, kw)
    if _compile_state_isinsert(compile_state) and stmt._select_names:
        assert not compile_state._has_multi_parameters
        _scan_insert_from_select_cols(compiler, stmt, compile_state, parameters, _getattr_col_key, _column_as_key, _col_bind_name, check_columns, values, toplevel, kw)
        use_insertmanyvalues = False
        use_sentinel_columns = None
    else:
        use_insertmanyvalues, use_sentinel_columns = _scan_cols(compiler, stmt, compile_state, parameters, _getattr_col_key, _column_as_key, _col_bind_name, check_columns, values, toplevel, kw)
    if parameters and stmt_parameter_tuples:
        check = set(parameters).intersection((_column_as_key(k) for k, v in stmt_parameter_tuples)).difference(check_columns)
        if check:
            raise exc.CompileError('Unconsumed column names: %s' % ', '.join(('%s' % (c,) for c in check)))
    is_default_metavalue_only = False
    if _compile_state_isinsert(compile_state) and compile_state._has_multi_parameters:
        assert not stmt._select_names
        multi_extended_values = _extend_values_for_multiparams(compiler, stmt, compile_state, cast('Sequence[_CrudParamElementStr]', values), cast('Callable[..., str]', _column_as_key), kw)
        return _CrudParams(values, multi_extended_values)
    elif not values and compiler.for_executemany and compiler.dialect.supports_default_metavalue:
        values = [(_as_dml_column(stmt.table.columns[0]), compiler.preparer.format_column(stmt.table.columns[0]), compiler.dialect.default_metavalue_token, ())]
        is_default_metavalue_only = True
    return _CrudParams(values, [], is_default_metavalue_only=is_default_metavalue_only, use_insertmanyvalues=use_insertmanyvalues, use_sentinel_columns=use_sentinel_columns)