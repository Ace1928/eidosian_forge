from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
class DataclassDumper(yaml.Dumper):

    def ignore_aliases(self, data):
        return super().ignore_aliases(data) or data is _fields.MISSING_PROP