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
def validate_email(value: str) -> tuple[str, str]:
    """Email address validation using [email-validator](https://pypi.org/project/email-validator/).

    Note:
        Note that:

        * Raw IP address (literal) domain parts are not allowed.
        * `"John Doe <local_part@domain.com>"` style "pretty" email addresses are processed.
        * Spaces are striped from the beginning and end of addresses, but no error is raised.
    """
    if email_validator is None:
        import_email_validator()
    if len(value) > MAX_EMAIL_LENGTH:
        raise PydanticCustomError('value_error', 'value is not a valid email address: {reason}', {'reason': f'Length must not exceed {MAX_EMAIL_LENGTH} characters'})
    m = pretty_email_regex.fullmatch(value)
    name: str | None = None
    if m:
        unquoted_name, quoted_name, value = m.groups()
        name = unquoted_name or quoted_name
    email = value.strip()
    try:
        parts = email_validator.validate_email(email, check_deliverability=False)
    except email_validator.EmailNotValidError as e:
        raise PydanticCustomError('value_error', 'value is not a valid email address: {reason}', {'reason': str(e.args[0])}) from e
    email = parts.normalized
    assert email is not None
    name = name or parts.local_part
    return (name, email)