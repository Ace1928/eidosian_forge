import copy
from dataclasses import is_dataclass
from itertools import zip_longest
from typing import TypeVar, Type, Optional, get_type_hints, Mapping, Any
from .config import Config
from .data import Data
from .dataclasses import get_default_value_for_field, create_instance, DefaultValueNotFoundError, get_fields
from .exceptions import (
from .types import (
def _build_value_for_union(union: Type, data: Any, config: Config) -> Any:
    types = extract_generic(union)
    if is_optional(union) and len(types) == 2:
        return _build_value(type_=types[0], data=data, config=config)
    union_matches = {}
    for inner_type in types:
        try:
            try:
                data = transform_value(type_hooks=config.type_hooks, cast=config.cast, target_type=inner_type, value=data)
            except Exception:
                continue
            value = _build_value(type_=inner_type, data=data, config=config)
            if is_instance(value, inner_type):
                if config.strict_unions_match:
                    union_matches[inner_type] = value
                else:
                    return value
        except DaciteError:
            pass
    if config.strict_unions_match:
        if len(union_matches) > 1:
            raise StrictUnionMatchError(union_matches)
        return union_matches.popitem()[1]
    if not config.check_types:
        return data
    raise UnionMatchError(field_type=union, value=data)