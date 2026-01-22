from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
class BaseModel(pydantic.BaseModel):
    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')
    else:

        @property
        @override
        def model_fields_set(self) -> set[str]:
            return self.__fields_set__

        class Config(pydantic.BaseConfig):
            extra: Any = pydantic.Extra.allow

    @override
    def __str__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(', ')})'

    @classmethod
    @override
    def construct(cls: Type[ModelT], _fields_set: set[str] | None=None, **values: object) -> ModelT:
        m = cls.__new__(cls)
        fields_values: dict[str, object] = {}
        config = get_model_config(cls)
        populate_by_name = config.allow_population_by_field_name if isinstance(config, _ConfigProtocol) else config.get('populate_by_name')
        if _fields_set is None:
            _fields_set = set()
        model_fields = get_model_fields(cls)
        for name, field in model_fields.items():
            key = field.alias
            if key is None or (key not in values and populate_by_name):
                key = name
            if key in values:
                fields_values[name] = _construct_field(value=values[key], field=field, key=key)
                _fields_set.add(name)
            else:
                fields_values[name] = field_get_default(field)
        _extra = {}
        for key, value in values.items():
            if key not in model_fields:
                if PYDANTIC_V2:
                    _extra[key] = value
                else:
                    _fields_set.add(key)
                    fields_values[key] = value
        object.__setattr__(m, '__dict__', fields_values)
        if PYDANTIC_V2:
            object.__setattr__(m, '__pydantic_private__', None)
            object.__setattr__(m, '__pydantic_extra__', _extra)
            object.__setattr__(m, '__pydantic_fields_set__', _fields_set)
        else:
            m._init_private_attributes()
            object.__setattr__(m, '__fields_set__', _fields_set)
        return m
    if not TYPE_CHECKING:
        model_construct = construct
    if not PYDANTIC_V2:

        @override
        def model_dump(self, *, mode: Literal['json', 'python'] | str='python', include: IncEx=None, exclude: IncEx=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, round_trip: bool=False, warnings: bool=True) -> dict[str, Any]:
            """Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump

            Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

            Args:
                mode: The mode in which `to_python` should run.
                    If mode is 'json', the dictionary will only contain JSON serializable types.
                    If mode is 'python', the dictionary may contain any Python objects.
                include: A list of fields to include in the output.
                exclude: A list of fields to exclude from the output.
                by_alias: Whether to use the field's alias in the dictionary key if defined.
                exclude_unset: Whether to exclude fields that are unset or None from the output.
                exclude_defaults: Whether to exclude fields that are set to their default value from the output.
                exclude_none: Whether to exclude fields that have a value of `None` from the output.
                round_trip: Whether to enable serialization and deserialization round-trip support.
                warnings: Whether to log warnings when invalid fields are encountered.

            Returns:
                A dictionary representation of the model.
            """
            if mode != 'python':
                raise ValueError('mode is only supported in Pydantic v2')
            if round_trip != False:
                raise ValueError('round_trip is only supported in Pydantic v2')
            if warnings != True:
                raise ValueError('warnings is only supported in Pydantic v2')
            return super().dict(include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

        @override
        def model_dump_json(self, *, indent: int | None=None, include: IncEx=None, exclude: IncEx=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, round_trip: bool=False, warnings: bool=True) -> str:
            """Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

            Generates a JSON representation of the model using Pydantic's `to_json` method.

            Args:
                indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
                include: Field(s) to include in the JSON output. Can take either a string or set of strings.
                exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
                by_alias: Whether to serialize using field aliases.
                exclude_unset: Whether to exclude fields that have not been explicitly set.
                exclude_defaults: Whether to exclude fields that have the default value.
                exclude_none: Whether to exclude fields that have a value of `None`.
                round_trip: Whether to use serialization/deserialization between JSON and class instance.
                warnings: Whether to show any warnings that occurred during serialization.

            Returns:
                A JSON string representation of the model.
            """
            if round_trip != False:
                raise ValueError('round_trip is only supported in Pydantic v2')
            if warnings != True:
                raise ValueError('warnings is only supported in Pydantic v2')
            return super().json(indent=indent, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)