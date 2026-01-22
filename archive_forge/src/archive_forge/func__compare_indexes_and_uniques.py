from __future__ import annotations
import contextlib
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import event
from sqlalchemy import inspect
from sqlalchemy import schema as sa_schema
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.sql import expression
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from sqlalchemy.util import OrderedSet
from .. import util
from ..ddl._autogen import is_index_sig
from ..ddl._autogen import is_uq_sig
from ..operations import ops
from ..util import sqla_compat
@comparators.dispatch_for('table')
def _compare_indexes_and_uniques(autogen_context: AutogenContext, modify_ops: ModifyTableOps, schema: Optional[str], tname: Union[quoted_name, str], conn_table: Optional[Table], metadata_table: Optional[Table]) -> None:
    inspector = autogen_context.inspector
    is_create_table = conn_table is None
    is_drop_table = metadata_table is None
    impl = autogen_context.migration_context.impl
    if metadata_table is not None:
        metadata_unique_constraints = {uq for uq in metadata_table.constraints if isinstance(uq, sa_schema.UniqueConstraint)}
        metadata_indexes = set(metadata_table.indexes)
    else:
        metadata_unique_constraints = set()
        metadata_indexes = set()
    conn_uniques = conn_indexes = frozenset()
    supports_unique_constraints = False
    unique_constraints_duplicate_unique_indexes = False
    if conn_table is not None:
        try:
            conn_uniques = inspector.get_unique_constraints(tname, schema=schema)
            supports_unique_constraints = True
        except NotImplementedError:
            pass
        except TypeError:
            pass
        else:
            conn_uniques = [uq for uq in conn_uniques if autogen_context.run_name_filters(uq['name'], 'unique_constraint', {'table_name': tname, 'schema_name': schema})]
            for uq in conn_uniques:
                if uq.get('duplicates_index'):
                    unique_constraints_duplicate_unique_indexes = True
        try:
            conn_indexes = inspector.get_indexes(tname, schema=schema)
        except NotImplementedError:
            pass
        else:
            conn_indexes = [ix for ix in conn_indexes if autogen_context.run_name_filters(ix['name'], 'index', {'table_name': tname, 'schema_name': schema})]
        if is_drop_table:
            conn_uniques = set()
        else:
            conn_uniques = {_make_unique_constraint(impl, uq_def, conn_table) for uq_def in conn_uniques}
        conn_indexes = {index for index in (_make_index(impl, ix, conn_table) for ix in conn_indexes) if index is not None}
    if unique_constraints_duplicate_unique_indexes:
        _correct_for_uq_duplicates_uix(conn_uniques, conn_indexes, metadata_unique_constraints, metadata_indexes, autogen_context.dialect, impl)
    impl.correct_for_autogen_constraints(conn_uniques, conn_indexes, metadata_unique_constraints, metadata_indexes)
    metadata_unique_constraints_sig = {impl._create_metadata_constraint_sig(uq) for uq in metadata_unique_constraints}
    metadata_indexes_sig = {impl._create_metadata_constraint_sig(ix) for ix in metadata_indexes}
    conn_unique_constraints = {impl._create_reflected_constraint_sig(uq) for uq in conn_uniques}
    conn_indexes_sig = {impl._create_reflected_constraint_sig(ix) for ix in conn_indexes}
    metadata_names = {cast(str, c.md_name_to_sql_name(autogen_context)): c for c in metadata_unique_constraints_sig.union(metadata_indexes_sig) if c.is_named}
    conn_uniques_by_name: Dict[sqla_compat._ConstraintName, _constraint_sig]
    conn_indexes_by_name: Dict[sqla_compat._ConstraintName, _constraint_sig]
    conn_uniques_by_name = {c.name: c for c in conn_unique_constraints}
    conn_indexes_by_name = {c.name: c for c in conn_indexes_sig}
    conn_names = {c.name: c for c in conn_unique_constraints.union(conn_indexes_sig) if sqla_compat.constraint_name_string(c.name)}
    doubled_constraints = {name: (conn_uniques_by_name[name], conn_indexes_by_name[name]) for name in set(conn_uniques_by_name).intersection(conn_indexes_by_name)}
    conn_uniques_by_sig = {uq.unnamed: uq for uq in conn_unique_constraints}
    metadata_uniques_by_sig = {uq.unnamed: uq for uq in metadata_unique_constraints_sig}
    unnamed_metadata_uniques = {uq.unnamed: uq for uq in metadata_unique_constraints_sig if not sqla_compat._constraint_is_named(uq.const, autogen_context.dialect)}

    def obj_added(obj: _constraint_sig):
        if is_index_sig(obj):
            if autogen_context.run_object_filters(obj.const, obj.name, 'index', False, None):
                modify_ops.ops.append(ops.CreateIndexOp.from_index(obj.const))
                log.info("Detected added index '%r' on '%s'", obj.name, obj.column_names)
        elif is_uq_sig(obj):
            if not supports_unique_constraints:
                return
            if is_create_table or is_drop_table:
                return
            if autogen_context.run_object_filters(obj.const, obj.name, 'unique_constraint', False, None):
                modify_ops.ops.append(ops.AddConstraintOp.from_constraint(obj.const))
                log.info("Detected added unique constraint %r on '%s'", obj.name, obj.column_names)
        else:
            assert False

    def obj_removed(obj: _constraint_sig):
        if is_index_sig(obj):
            if obj.is_unique and (not supports_unique_constraints):
                return
            if autogen_context.run_object_filters(obj.const, obj.name, 'index', True, None):
                modify_ops.ops.append(ops.DropIndexOp.from_index(obj.const))
                log.info('Detected removed index %r on %r', obj.name, tname)
        elif is_uq_sig(obj):
            if is_create_table or is_drop_table:
                return
            if autogen_context.run_object_filters(obj.const, obj.name, 'unique_constraint', True, None):
                modify_ops.ops.append(ops.DropConstraintOp.from_constraint(obj.const))
                log.info('Detected removed unique constraint %r on %r', obj.name, tname)
        else:
            assert False

    def obj_changed(old: _constraint_sig, new: _constraint_sig, msg: str):
        if is_index_sig(old):
            assert is_index_sig(new)
            if autogen_context.run_object_filters(new.const, new.name, 'index', False, old.const):
                log.info('Detected changed index %r on %r: %s', old.name, tname, msg)
                modify_ops.ops.append(ops.DropIndexOp.from_index(old.const))
                modify_ops.ops.append(ops.CreateIndexOp.from_index(new.const))
        elif is_uq_sig(old):
            assert is_uq_sig(new)
            if autogen_context.run_object_filters(new.const, new.name, 'unique_constraint', False, old.const):
                log.info('Detected changed unique constraint %r on %r: %s', old.name, tname, msg)
                modify_ops.ops.append(ops.DropConstraintOp.from_constraint(old.const))
                modify_ops.ops.append(ops.AddConstraintOp.from_constraint(new.const))
        else:
            assert False
    for removed_name in sorted(set(conn_names).difference(metadata_names)):
        conn_obj = conn_names[removed_name]
        if is_uq_sig(conn_obj) and conn_obj.unnamed in unnamed_metadata_uniques:
            continue
        elif removed_name in doubled_constraints:
            conn_uq, conn_idx = doubled_constraints[removed_name]
            if all((conn_idx.unnamed != meta_idx.unnamed for meta_idx in metadata_indexes_sig)) and conn_uq.unnamed not in metadata_uniques_by_sig:
                obj_removed(conn_uq)
                obj_removed(conn_idx)
        else:
            obj_removed(conn_obj)
    for existing_name in sorted(set(metadata_names).intersection(conn_names)):
        metadata_obj = metadata_names[existing_name]
        if existing_name in doubled_constraints:
            conn_uq, conn_idx = doubled_constraints[existing_name]
            if is_index_sig(metadata_obj):
                conn_obj = conn_idx
            else:
                conn_obj = conn_uq
        else:
            conn_obj = conn_names[existing_name]
        if type(conn_obj) != type(metadata_obj):
            obj_removed(conn_obj)
            obj_added(metadata_obj)
        else:
            comparison = metadata_obj.compare_to_reflected(conn_obj)
            if comparison.is_different:
                obj_changed(conn_obj, metadata_obj, comparison.message)
            elif comparison.is_skip:
                thing = 'index' if is_index_sig(conn_obj) else 'unique constraint'
                log.info('Cannot compare %s %r, assuming equal and skipping. %s', thing, conn_obj.name, comparison.message)
            else:
                assert comparison.is_equal
    for added_name in sorted(set(metadata_names).difference(conn_names)):
        obj = metadata_names[added_name]
        obj_added(obj)
    for uq_sig in unnamed_metadata_uniques:
        if uq_sig not in conn_uniques_by_sig:
            obj_added(unnamed_metadata_uniques[uq_sig])