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
def _build_pretty_email_regex() -> re.Pattern[str]:
    name_chars = "[\\w!#$%&\\'*+\\-/=?^_`{|}~]"
    unquoted_name_group = f'((?:{name_chars}+\\s+)*{name_chars}+)'
    quoted_name_group = '"((?:[^"]|\\")+)"'
    email_group = '<\\s*(.+)\\s*>'
    return re.compile(f'\\s*(?:{unquoted_name_group}|{quoted_name_group})?\\s*{email_group}\\s*')