from functools import partial
from typing import Callable
from . import converters, exceptions, filters, setters, validators
from ._cmp import cmp_using
from ._compat import Protocol
from ._config import get_run_validators, set_run_validators
from ._funcs import asdict, assoc, astuple, evolve, has, resolve_types
from ._make import (
from ._next_gen import define, field, frozen, mutable
from ._version_info import VersionInfo
class AttrsInstance(Protocol):
    pass