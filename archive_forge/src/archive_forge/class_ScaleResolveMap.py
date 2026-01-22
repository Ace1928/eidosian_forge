from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ScaleResolveMap(VegaLiteSchema):
    """ScaleResolveMap schema wrapper

    Parameters
    ----------

    angle : :class:`ResolveMode`, Literal['independent', 'shared']

    color : :class:`ResolveMode`, Literal['independent', 'shared']

    fill : :class:`ResolveMode`, Literal['independent', 'shared']

    fillOpacity : :class:`ResolveMode`, Literal['independent', 'shared']

    opacity : :class:`ResolveMode`, Literal['independent', 'shared']

    radius : :class:`ResolveMode`, Literal['independent', 'shared']

    shape : :class:`ResolveMode`, Literal['independent', 'shared']

    size : :class:`ResolveMode`, Literal['independent', 'shared']

    stroke : :class:`ResolveMode`, Literal['independent', 'shared']

    strokeDash : :class:`ResolveMode`, Literal['independent', 'shared']

    strokeOpacity : :class:`ResolveMode`, Literal['independent', 'shared']

    strokeWidth : :class:`ResolveMode`, Literal['independent', 'shared']

    theta : :class:`ResolveMode`, Literal['independent', 'shared']

    x : :class:`ResolveMode`, Literal['independent', 'shared']

    xOffset : :class:`ResolveMode`, Literal['independent', 'shared']

    y : :class:`ResolveMode`, Literal['independent', 'shared']

    yOffset : :class:`ResolveMode`, Literal['independent', 'shared']

    """
    _schema = {'$ref': '#/definitions/ScaleResolveMap'}

    def __init__(self, angle: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, color: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, fill: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, fillOpacity: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, opacity: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, radius: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, shape: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, size: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, stroke: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, strokeDash: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, strokeOpacity: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, strokeWidth: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, theta: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, x: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, xOffset: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, y: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, yOffset: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, **kwds):
        super(ScaleResolveMap, self).__init__(angle=angle, color=color, fill=fill, fillOpacity=fillOpacity, opacity=opacity, radius=radius, shape=shape, size=size, stroke=stroke, strokeDash=strokeDash, strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, theta=theta, x=x, xOffset=xOffset, y=y, yOffset=yOffset, **kwds)