import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
def _parse_spec_collection(self, obj: Any, to_type: Type[T]) -> IndexedOrderedDict[str, T]:
    res: IndexedOrderedDict[str, T] = IndexedOrderedDict()
    if obj is None:
        return res
    aot(isinstance(obj, List), 'Spec collection must be a list')
    for v in obj:
        s = self._parse_spec(v, to_type)
        aot(s.name not in res, KeyError(f'Duplicated key {s.name}'))
        res[s.name] = s
    return res