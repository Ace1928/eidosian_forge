from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@classmethod
def _cls_checks(cls, format: str, raises: _RaisesType=()) -> typing.Callable[[_F], _F]:

    def _checks(func: _F) -> _F:
        cls.checkers[format] = (func, raises)
        return func
    return _checks