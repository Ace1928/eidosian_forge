import binascii
import math
import struct
import sys
import pytest
from shapely import wkt
from shapely.geometry import Point
from shapely.geos import geos_version
from shapely.tests.legacy.conftest import shapely20_todo
from shapely.wkb import dump, dumps, load, loads
@pytest.mark.xfail(geos_version < (3, 9, 0) and (not (geos_version < (3, 8, 0) and sys.platform == 'darwin')), reason='GEOS >= 3.9.0 is required')
def test_point_z_empty():
    g = wkt.loads('POINT Z EMPTY')
    assert g.wkb_hex == hostorder('BIddd', '0101000080000000000000F87F000000000000F87F000000000000F87F')