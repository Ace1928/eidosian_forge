import collections
import textwrap
from dataclasses import dataclass, field
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Callable
from xarray.core import duck_array_ops
from xarray.core.types import Dims, Self
Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python xarray/util/generate_aggregations.py
    pytest --doctest-modules xarray/core/_aggregations.py --accept || true
    pytest --doctest-modules xarray/core/_aggregations.py

This requires [pytest-accept](https://github.com/max-sixty/pytest-accept).
The second run of pytest is deliberate, since the first will return an error
while replacing the doctests.

