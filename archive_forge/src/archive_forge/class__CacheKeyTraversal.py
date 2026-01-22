from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
class _CacheKeyTraversal(HasTraversalDispatch):
    visit_has_cache_key = visit_clauseelement = CALL_GEN_CACHE_KEY
    visit_clauseelement_list = InternalTraversal.dp_clauseelement_list
    visit_annotations_key = InternalTraversal.dp_annotations_key
    visit_clauseelement_tuple = InternalTraversal.dp_clauseelement_tuple
    visit_memoized_select_entities = InternalTraversal.dp_memoized_select_entities
    visit_string = visit_boolean = visit_operator = visit_plain_obj = CACHE_IN_PLACE
    visit_statement_hint_list = CACHE_IN_PLACE
    visit_type = STATIC_CACHE_KEY
    visit_anon_name = ANON_NAME
    visit_propagate_attrs = PROPAGATE_ATTRS

    def visit_with_context_options(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return tuple(((fn.__code__, c_key) for fn, c_key in obj))

    def visit_inspectable(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, inspect(obj)._gen_cache_key(anon_map, bindparams))

    def visit_string_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return tuple(obj)

    def visit_multi(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, obj._gen_cache_key(anon_map, bindparams) if isinstance(obj, HasCacheKey) else obj)

    def visit_multi_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) if isinstance(elem, HasCacheKey) else elem for elem in obj)))

    def visit_has_cache_key_tuples(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (attrname, tuple((tuple((elem._gen_cache_key(anon_map, bindparams) for elem in tup_elem)) for tup_elem in obj)))

    def visit_has_cache_key_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) for elem in obj)))

    def visit_executable_options(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) for elem in obj if elem._is_has_cache_key)))

    def visit_inspectable_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return self.visit_has_cache_key_list(attrname, [inspect(o) for o in obj], parent, anon_map, bindparams)

    def visit_clauseelement_tuples(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return self.visit_has_cache_key_tuples(attrname, obj, parent, anon_map, bindparams)

    def visit_fromclause_ordered_set(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (attrname, tuple([elem._gen_cache_key(anon_map, bindparams) for elem in obj]))

    def visit_clauseelement_unordered_set(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        cache_keys = [elem._gen_cache_key(anon_map, bindparams) for elem in obj]
        return (attrname, tuple(sorted(cache_keys)))

    def visit_named_ddl_element(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, obj.name)

    def visit_prefix_sequence(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (attrname, tuple([(clause._gen_cache_key(anon_map, bindparams), strval) for clause, strval in obj]))

    def visit_setup_join_tuple(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return tuple(((target._gen_cache_key(anon_map, bindparams), onclause._gen_cache_key(anon_map, bindparams) if onclause is not None else None, from_._gen_cache_key(anon_map, bindparams) if from_ is not None else None, tuple([(key, flags[key]) for key in sorted(flags)])) for target, onclause, from_, flags in obj))

    def visit_table_hint_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (attrname, tuple([(clause._gen_cache_key(anon_map, bindparams), dialect_name, text) for (clause, dialect_name), text in obj.items()]))

    def visit_plain_dict(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple([(key, obj[key]) for key in sorted(obj)]))

    def visit_dialect_options(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple(((dialect_name, tuple([(key, obj[dialect_name][key]) for key in sorted(obj[dialect_name])])) for dialect_name in sorted(obj))))

    def visit_string_clauseelement_dict(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple(((key, obj[key]._gen_cache_key(anon_map, bindparams)) for key in sorted(obj))))

    def visit_string_multi_dict(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple(((key, value._gen_cache_key(anon_map, bindparams) if isinstance(value, HasCacheKey) else value) for key, value in [(key, obj[key]) for key in sorted(obj)])))

    def visit_fromclause_canonical_column_collection(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple((col._gen_cache_key(anon_map, bindparams) for k, col, _ in obj._collection)))

    def visit_unknown_structure(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        anon_map[NO_CACHE] = True
        return ()

    def visit_dml_ordered_values(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple(((key._gen_cache_key(anon_map, bindparams) if hasattr(key, '__clause_element__') else key, value._gen_cache_key(anon_map, bindparams)) for key, value in obj)))

    def visit_dml_values(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (attrname, tuple(((k._gen_cache_key(anon_map, bindparams) if hasattr(k, '__clause_element__') else k, obj[k]._gen_cache_key(anon_map, bindparams)) for k in obj)))

    def visit_dml_multi_values(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        anon_map[NO_CACHE] = True
        return ()