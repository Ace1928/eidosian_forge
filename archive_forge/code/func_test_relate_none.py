from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('g1, g2', [(point, None), (None, point), (None, None)])
def test_relate_none(g1, g2):
    assert shapely.relate(g1, g2) is None