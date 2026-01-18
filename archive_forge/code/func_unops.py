from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Optional
from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, Callable, overload
from xarray.core import nputils, ops
from xarray.core.types import (
def unops() -> list[OpsType]:
    return [([(None, None)], required_method_unary, {}), (UNARY_OPS, template_unary, {}), (OTHER_UNARY_METHODS, template_other_unary, {})]