from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class GeometryCollection(Geometry):
    """GeometryCollection schema wrapper
    Geometry Collection https://tools.ietf.org/html/rfc7946#section-3.1.8

    Parameters
    ----------

    geometries : Sequence[dict, :class:`Point`, :class:`Polygon`, :class:`Geometry`, :class:`LineString`, :class:`MultiPoint`, :class:`MultiPolygon`, :class:`MultiLineString`, :class:`GeometryCollection`]

    type : str
        Specifies the type of GeoJSON object.
    bbox : :class:`BBox`, Sequence[float]
        Bounding box of the coordinate range of the object's Geometries, Features, or
        Feature Collections. https://tools.ietf.org/html/rfc7946#section-5
    """
    _schema = {'$ref': '#/definitions/GeometryCollection'}

    def __init__(self, geometries: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, type: Union[str, UndefinedType]=Undefined, bbox: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, **kwds):
        super(GeometryCollection, self).__init__(geometries=geometries, type=type, bbox=bbox, **kwds)