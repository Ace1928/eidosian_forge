from binascii import hexlify
from collections.abc import MutableMapping
from collections import OrderedDict
from enum import Enum
import itertools
from json import JSONEncoder
from warnings import warn
from fiona.errors import FionaDeprecationWarning
class OGRGeometryType(Enum):
    Unknown = 0
    Point = 1
    LineString = 2
    Polygon = 3
    MultiPoint = 4
    MultiLineString = 5
    MultiPolygon = 6
    GeometryCollection = 7
    CircularString = 8
    CompoundCurve = 9
    CurvePolygon = 10
    MultiCurve = 11
    MultiSurface = 12
    Curve = 13
    Surface = 14
    PolyhedralSurface = 15
    TIN = 16
    Triangle = 17
    NONE = 100
    LinearRing = 101
    CircularStringZ = 1008
    CompoundCurveZ = 1009
    CurvePolygonZ = 1010
    MultiCurveZ = 1011
    MultiSurfaceZ = 1012
    CurveZ = 1013
    SurfaceZ = 1014
    PolyhedralSurfaceZ = 1015
    TINZ = 1016
    TriangleZ = 1017
    PointM = 2001
    LineStringM = 2002
    PolygonM = 2003
    MultiPointM = 2004
    MultiLineStringM = 2005
    MultiPolygonM = 2006
    GeometryCollectionM = 2007
    CircularStringM = 2008
    CompoundCurveM = 2009
    CurvePolygonM = 2010
    MultiCurveM = 2011
    MultiSurfaceM = 2012
    CurveM = 2013
    SurfaceM = 2014
    PolyhedralSurfaceM = 2015
    TINM = 2016
    TriangleM = 2017
    PointZM = 3001
    LineStringZM = 3002
    PolygonZM = 3003
    MultiPointZM = 3004
    MultiLineStringZM = 3005
    MultiPolygonZM = 3006
    GeometryCollectionZM = 3007
    CircularStringZM = 3008
    CompoundCurveZM = 3009
    CurvePolygonZM = 3010
    MultiCurveZM = 3011
    MultiSurfaceZM = 3012
    CurveZM = 3013
    SurfaceZM = 3014
    PolyhedralSurfaceZM = 3015
    TINZM = 3016
    TriangleZM = 3017
    Point25D = 2147483649
    LineString25D = 2147483650
    Polygon25D = 2147483651
    MultiPoint25D = 2147483652
    MultiLineString25D = 2147483653
    MultiPolygon25D = 2147483654
    GeometryCollection25D = 2147483655