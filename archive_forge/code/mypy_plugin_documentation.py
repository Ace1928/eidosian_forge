from __future__ import annotations
from collections.abc import Iterable
from typing import Final, TYPE_CHECKING, Callable
import numpy as np
Handle all import-based overrides.

            * Import platform-specific extended-precision `numpy.number`
              subclasses (*e.g.* `numpy.float96`, `numpy.float128` and
              `numpy.complex256`).
            * Import the appropriate `ctypes` equivalent to `numpy.intp`.

            