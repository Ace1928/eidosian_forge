from __future__ import annotations
from typing import Callable, Generator, List, TypeVar, Union, Tuple, Any, Sequence
from typing_extensions import Literal, Never
import numpy as np
from numpy.typing import ArrayLike
def _ensure_not_reachable(__arg: Never):
    """
    Ensure that a code path is not reachable, like typing_extension.assert_never.

    This doesn't raise an exception so that we are forced to manually
    raise a more user friendly exception afterwards.
    """
    ...