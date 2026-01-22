from __future__ import annotations as _annotations
import dataclasses as _dataclasses
import re
from importlib.metadata import version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import TYPE_CHECKING, Any
from pydantic_core import MultiHostUrl, PydanticCustomError, Url, core_schema
from typing_extensions import Annotated, TypeAlias
from ._internal import _fields, _repr, _schema_generation_shared
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler
from .json_schema import JsonSchemaValue
from pydantic import BaseModel, HttpUrl, ValidationError
from pydantic import BaseModel, HttpUrl
from pydantic import (
class IPvAnyAddress:
    """Validate an IPv4 or IPv6 address.

    ```py
    from pydantic import BaseModel
    from pydantic.networks import IPvAnyAddress

    class IpModel(BaseModel):
        ip: IPvAnyAddress

    print(IpModel(ip='127.0.0.1'))
    #> ip=IPv4Address('127.0.0.1')

    try:
        IpModel(ip='http://www.example.com')
    except ValueError as e:
        print(e.errors())
        '''
        [
            {
                'type': 'ip_any_address',
                'loc': ('ip',),
                'msg': 'value is not a valid IPv4 or IPv6 address',
                'input': 'http://www.example.com',
            }
        ]
        '''
    ```
    """
    __slots__ = ()

    def __new__(cls, value: Any) -> IPv4Address | IPv6Address:
        """Validate an IPv4 or IPv6 address."""
        try:
            return IPv4Address(value)
        except ValueError:
            pass
        try:
            return IPv6Address(value)
        except ValueError:
            raise PydanticCustomError('ip_any_address', 'value is not a valid IPv4 or IPv6 address')

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = {}
        field_schema.update(type='string', format='ipvanyaddress')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

    @classmethod
    def _validate(cls, __input_value: Any) -> IPv4Address | IPv6Address:
        return cls(__input_value)