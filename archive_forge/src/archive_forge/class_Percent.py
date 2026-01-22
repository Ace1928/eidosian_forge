from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar, Union
from ...util.deprecation import deprecated
from .bases import (
from .primitive import Float, Int
from .singletons import Intrinsic, Undefined
class Percent(Float):
    """ Accept floating point percentage values.

    ``Percent`` can be useful and semantically meaningful for specifying
    things like alpha values and extents.

    Args:
        default (float, optional) :
            A default value for attributes created from this property to have.

        help (str or None, optional) :
            A documentation string for this property. It will be automatically
            used by the :ref:`bokeh.sphinxext.bokeh_prop` extension when
            generating Spinx documentation. (default: None)

    Example:

        .. code-block:: python

            >>> class PercentModel(HasProps):
            ...     prop = Percent()
            ...

            >>> m = PercentModel()

            >>> m.prop = 0.0

            >>> m.prop = 0.2

            >>> m.prop = 1.0

            >>> m.prop = -2  # ValueError !!

            >>> m.prop = 5   # ValueError !!

    """

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if 0.0 <= value <= 1.0:
            return
        msg = '' if not detail else f'expected a value in range [0, 1], got {value!r}'
        raise ValueError(msg)