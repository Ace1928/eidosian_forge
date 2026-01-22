from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def _make_loader(cls: Type[Any]) -> Type[yaml.Loader]:
    import yaml

    class DataclassLoader(yaml.Loader):
        pass
    contained_types = list(_get_contained_special_types_from_type(cls))
    contained_type_names = list(map(lambda cls: cls.__name__, contained_types))
    assert len(set(contained_type_names)) == len(contained_type_names), f'Contained dataclass type names must all be unique, but got {contained_type_names}'

    def make_dataclass_constructor(typ: Type[Any]):
        return lambda loader, node: typ(**loader.construct_mapping(node))

    def make_enum_constructor(typ: Type[enum.Enum]):
        return lambda loader, node: typ[loader.construct_python_str(node)]
    for typ, name in zip(contained_types, contained_type_names):
        if dataclasses.is_dataclass(typ):
            DataclassLoader.add_constructor(tag=DATACLASS_YAML_TAG_PREFIX + name, constructor=make_dataclass_constructor(typ))
        elif issubclass(typ, enum.Enum):
            DataclassLoader.add_constructor(tag=ENUM_YAML_TAG_PREFIX + name, constructor=make_enum_constructor(typ))
        else:
            assert False
    DataclassLoader.add_constructor(tag=MISSING_YAML_TAG_PREFIX, constructor=lambda *_unused: _fields.MISSING_PROP)
    return DataclassLoader