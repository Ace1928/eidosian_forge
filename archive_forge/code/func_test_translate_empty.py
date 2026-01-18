import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_translate_empty(self):
    tls = affinity.translate(load_wkt('LINESTRING EMPTY'))
    els = load_wkt('LINESTRING EMPTY')
    self.assertTrue(tls.equals(els))
    assert tls.equals(els)