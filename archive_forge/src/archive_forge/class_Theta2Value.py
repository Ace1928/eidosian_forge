import sys
from . import core
import pandas as pd
from altair.utils.schemapi import Undefined, UndefinedType, with_property_setters
from altair.utils import parse_shorthand
from typing import Any, overload, Sequence, List, Literal, Union, Optional
from typing import Dict as TypingDict
@with_property_setters
class Theta2Value(ValueChannelMixin, core.PositionValueDef):
    """Theta2Value schema wrapper
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Parameters
    ----------

    value : str, dict, float, :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _class_is_valid_at_instantiation = False
    _encoding_name = 'theta2'

    def __init__(self, value, **kwds):
        super(Theta2Value, self).__init__(value=value, **kwds)