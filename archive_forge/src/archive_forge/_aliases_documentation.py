from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
from types import ModuleType
import inspect
from ._helpers import _check_device, _is_numpy_array, array_namespace

    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Note that outside of this function, this compat library does not yet fully
    support complex numbers.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    