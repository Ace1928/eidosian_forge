from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class GraticuleParams(VegaLiteSchema):
    """GraticuleParams schema wrapper

    Parameters
    ----------

    extent : :class:`Vector2Vector2number`, Sequence[Sequence[float], :class:`Vector2number`]
        Sets both the major and minor extents to the same values.
    extentMajor : :class:`Vector2Vector2number`, Sequence[Sequence[float], :class:`Vector2number`]
        The major extent of the graticule as a two-element array of coordinates.
    extentMinor : :class:`Vector2Vector2number`, Sequence[Sequence[float], :class:`Vector2number`]
        The minor extent of the graticule as a two-element array of coordinates.
    precision : float
        The precision of the graticule in degrees.

        **Default value:** ``2.5``
    step : Sequence[float], :class:`Vector2number`
        Sets both the major and minor step angles to the same values.
    stepMajor : Sequence[float], :class:`Vector2number`
        The major step angles of the graticule.

        **Default value:** ``[90, 360]``
    stepMinor : Sequence[float], :class:`Vector2number`
        The minor step angles of the graticule.

        **Default value:** ``[10, 10]``
    """
    _schema = {'$ref': '#/definitions/GraticuleParams'}

    def __init__(self, extent: Union['SchemaBase', Sequence[Union['SchemaBase', Sequence[float]]], UndefinedType]=Undefined, extentMajor: Union['SchemaBase', Sequence[Union['SchemaBase', Sequence[float]]], UndefinedType]=Undefined, extentMinor: Union['SchemaBase', Sequence[Union['SchemaBase', Sequence[float]]], UndefinedType]=Undefined, precision: Union[float, UndefinedType]=Undefined, step: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, stepMajor: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, stepMinor: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, **kwds):
        super(GraticuleParams, self).__init__(extent=extent, extentMajor=extentMajor, extentMinor=extentMinor, precision=precision, step=step, stepMajor=stepMajor, stepMinor=stepMinor, **kwds)