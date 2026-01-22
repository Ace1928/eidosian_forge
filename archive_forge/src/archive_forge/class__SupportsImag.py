from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import (
import numpy as np
class _SupportsImag(Protocol[_T_co]):

    @property
    def imag(self) -> _T_co:
        ...