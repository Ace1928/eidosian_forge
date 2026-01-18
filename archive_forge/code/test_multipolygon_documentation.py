import numpy as np
import pytest
from shapely import MultiPolygon, Polygon
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
A list of multipolygons is not a valid multipolygon ctor argument