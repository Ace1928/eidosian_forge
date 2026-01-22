import copy
from dataclasses import is_dataclass
from itertools import zip_longest
from typing import TypeVar, Type, Optional, get_type_hints, Mapping, Any
from .config import Config
from .data import Data
from .dataclasses import get_default_value_for_field, create_instance, DefaultValueNotFoundError, get_fields
from .exceptions import (
from .types import (
def _build_value_for_collection(collection: Type, data: Any, config: Config) -> Any:
    data_type = data.__class__
    if is_instance(data, Mapping):
        item_type = extract_generic(collection, defaults=(Any, Any))[1]
        return data_type(((key, _build_value(type_=item_type, data=value, config=config)) for key, value in data.items()))
    elif is_instance(data, tuple):
        types = extract_generic(collection)
        if len(types) == 2 and types[1] == Ellipsis:
            return data_type((_build_value(type_=types[0], data=item, config=config) for item in data))
        return data_type((_build_value(type_=type_, data=item, config=config) for item, type_ in zip_longest(data, types)))
    item_type = extract_generic(collection, defaults=(Any,))[0]
    return data_type((_build_value(type_=item_type, data=item, config=config) for item in data))