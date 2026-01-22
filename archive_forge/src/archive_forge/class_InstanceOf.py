from __future__ import annotations as _annotations
import dataclasses
import sys
from functools import partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, cast, overload
from pydantic_core import core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import GetCoreSchemaHandler as _GetCoreSchemaHandler
from ._internal import _core_metadata, _decorators, _generics, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
from .errors import PydanticUserError
@dataclasses.dataclass(**_internal_dataclass.slots_true)
class InstanceOf:
    '''Generic type for annotating a type that is an instance of a given class.

        Example:
            ```py
            from pydantic import BaseModel, InstanceOf

            class Foo:
                ...

            class Bar(BaseModel):
                foo: InstanceOf[Foo]

            Bar(foo=Foo())
            try:
                Bar(foo=42)
            except ValidationError as e:
                print(e)
                """
                [
                │   {
                │   │   'type': 'is_instance_of',
                │   │   'loc': ('foo',),
                │   │   'msg': 'Input should be an instance of Foo',
                │   │   'input': 42,
                │   │   'ctx': {'class': 'Foo'},
                │   │   'url': 'https://errors.pydantic.dev/0.38.0/v/is_instance_of'
                │   }
                ]
                """
            ```
        '''

    @classmethod
    def __class_getitem__(cls, item: AnyType) -> AnyType:
        return Annotated[item, cls()]

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        from pydantic import PydanticSchemaGenerationError
        instance_of_schema = core_schema.is_instance_schema(_generics.get_origin(source) or source)
        try:
            original_schema = handler(source)
        except PydanticSchemaGenerationError:
            return instance_of_schema
        else:
            instance_of_schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(function=lambda v, h: h(v), schema=original_schema)
            return core_schema.json_or_python_schema(python_schema=instance_of_schema, json_schema=original_schema)
    __hash__ = object.__hash__