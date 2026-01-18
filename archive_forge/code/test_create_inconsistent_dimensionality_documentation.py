import pytest
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon, shape
from shapely.geos import geos_version

When a "context" passed to shape/asShape has a coordinate
which is missing a dimension we should raise a descriptive error.

When we use mixed dimensions in a WKT geometry, the parser strips
any dimension which is not present in every coordinate.
