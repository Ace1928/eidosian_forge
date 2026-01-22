from __future__ import annotations
from .. import ufunc
from .._utils import set_module
from typing import TYPE_CHECKING, final
from ._nested_sequence import (
from ._nbit import (
from ._char_codes import (
from ._scalars import (
from ._shape import (
from ._dtype_like import (
from ._array_like import (
@final
@set_module('numpy.typing')
class NBitBase:
    """
    A type representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    .. versionadded:: 1.20

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating
    a function that takes a float and integer of arbitrary precision
    as arguments and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import TypeVar, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> T1 = TypeVar("T1", bound=npt.NBitBase)
        >>> T2 = TypeVar("T2", bound=npt.NBitBase)

        >>> def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None:
        allowed_names = {'NBitBase', '_256Bit', '_128Bit', '_96Bit', '_80Bit', '_64Bit', '_32Bit', '_16Bit', '_8Bit'}
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()