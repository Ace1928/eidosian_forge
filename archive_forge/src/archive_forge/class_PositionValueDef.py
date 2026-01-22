from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class PositionValueDef(PolarDef, Position2Def, PositionDef):
    """PositionValueDef schema wrapper
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Parameters
    ----------

    value : str, dict, float, :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/PositionValueDef'}

    def __init__(self, value: Union[str, dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(PositionValueDef, self).__init__(value=value, **kwds)