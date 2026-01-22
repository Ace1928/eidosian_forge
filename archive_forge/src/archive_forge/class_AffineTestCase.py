import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
class AffineTestCase(unittest.TestCase):

    def test_affine_params(self):
        g = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
        with pytest.raises(TypeError):
            affinity.affine_transform(g, None)
        with pytest.raises(ValueError):
            affinity.affine_transform(g, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        with pytest.raises(AttributeError):
            affinity.affine_transform(None, [1, 2, 3, 4, 5, 6])

    def test_affine_geom_types(self):
        matrix2d = (1, 0, 0, 1, 0, 0)
        matrix3d = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
        empty2d = load_wkt('MULTIPOLYGON EMPTY')
        assert affinity.affine_transform(empty2d, matrix2d).is_empty

        def test_geom(g2, g3=None):
            assert not g2.has_z
            a2 = affinity.affine_transform(g2, matrix2d)
            assert not a2.has_z
            assert g2.equals(a2)
            if g3 is not None:
                assert g3.has_z
                a3 = affinity.affine_transform(g3, matrix3d)
                assert a3.has_z
                assert g3.equals(a3)
            return
        pt2d = load_wkt('POINT(12.3 45.6)')
        pt3d = load_wkt('POINT(12.3 45.6 7.89)')
        test_geom(pt2d, pt3d)
        ls2d = load_wkt('LINESTRING(0.9 3.4, 0.7 2, 2.5 2.7)')
        ls3d = load_wkt('LINESTRING(0.9 3.4 3.3, 0.7 2 2.3, 2.5 2.7 5.5)')
        test_geom(ls2d, ls3d)
        lr2d = load_wkt('LINEARRING(0.9 3.4, 0.7 2, 2.5 2.7, 0.9 3.4)')
        lr3d = load_wkt('LINEARRING(0.9 3.4 3.3, 0.7 2 2.3, 2.5 2.7 5.5, 0.9 3.4 3.3)')
        test_geom(lr2d, lr3d)
        test_geom(load_wkt('POLYGON((0.9 2.3, 0.5 1.1, 2.4 0.8, 0.9 2.3), (1.1 1.7, 0.9 1.3, 1.4 1.2, 1.1 1.7), (1.6 1.3, 1.7 1, 1.9 1.1, 1.6 1.3))'))
        test_geom(load_wkt('MULTIPOINT ((-300 300), (700 300), (-800 -1100), (200 -300))'))
        test_geom(load_wkt('MULTILINESTRING((0 0, -0.7 -0.7, 0.6 -1), (-0.5 0.5, 0.7 0.6, 0 -0.6))'))
        test_geom(load_wkt('MULTIPOLYGON(((900 4300, -1100 -400, 900 -800, 900 4300)), ((1200 4300, 2300 4400, 1900 1000, 1200 4300)))'))
        test_geom(load_wkt('GEOMETRYCOLLECTION(POINT(20 70), POLYGON((60 70, 13 35, 60 -30, 60 70)), LINESTRING(60 70, 50 100, 80 100))'))

    def test_affine_2d(self):
        g = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
        expected2d = load_wkt('LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)')
        matrix2d = (2, 0, 0, 2.5, -5, 4.1)
        a2 = affinity.affine_transform(g, matrix2d)
        assert a2.equals_exact(expected2d, 1e-06)
        assert not a2.has_z
        matrix3d = (2, 0, 0, 0, 2.5, 0, 0, 0, 10, -5, 4.1, 100)
        a3 = affinity.affine_transform(g, matrix3d)
        assert a3.equals_exact(expected2d, 1e-06)
        assert not a3.has_z

    def test_affine_3d(self):
        g2 = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
        g3 = load_wkt('LINESTRING(2.4 4.1 100.2, 2.4 3 132.8, 3 3 128.6)')
        matrix2d = (2, 0, 0, 2.5, -5, 4.1)
        matrix3d = (2, 0, 0, 0, 2.5, 0, 0, 0, 0.3048, -5, 4.1, 100)
        a22 = affinity.affine_transform(g2, matrix2d)
        a23 = affinity.affine_transform(g2, matrix3d)
        a32 = affinity.affine_transform(g3, matrix2d)
        a33 = affinity.affine_transform(g3, matrix3d)
        assert not a22.has_z
        assert not a23.has_z
        assert a32.has_z
        assert a33.has_z
        expected2d = load_wkt('LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)')
        expected3d = load_wkt('LINESTRING(-0.2 14.35 130.54096, -0.2 11.6 140.47744, 1 11.6 139.19728)')
        expected32 = load_wkt('LINESTRING(-0.2 14.35 100.2, -0.2 11.6 132.8, 1 11.6 128.6)')
        assert a22.equals_exact(expected2d, 1e-06)
        assert a23.equals_exact(expected2d, 1e-06)
        for a, e in zip(a32.coords, expected32.coords):
            for ap, ep in zip(a, e):
                self.assertAlmostEqual(ap, ep)
        for a, e in zip(a33.coords, expected3d.coords):
            for ap, ep in zip(a, e):
                self.assertAlmostEqual(ap, ep)