from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class MathML(MathText):
    """ Render mathematical content using `MathML <https://www.w3.org/Math/>`_
    notation.

    See :ref:`ug_styling_mathtext` in the |user guide| for more information.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)