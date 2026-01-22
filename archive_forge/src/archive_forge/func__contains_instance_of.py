from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
def _contains_instance_of(seq: Sequence[object], cl: type) -> bool:
    return any((isinstance(e, cl) for e in seq))