import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
class TestOverlayWikiExample:

    def setup_method(self):
        self.layer_a = GeoDataFrame(geometry=[box(0, 2, 6, 6)])
        self.layer_b = GeoDataFrame(geometry=[box(4, 0, 10, 4)])
        self.intersection = GeoDataFrame(geometry=[box(4, 2, 6, 4)])
        self.union = GeoDataFrame(geometry=[box(4, 2, 6, 4), Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)]), Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)])])
        self.a_difference_b = GeoDataFrame(geometry=[Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)])])
        self.b_difference_a = GeoDataFrame(geometry=[Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)])])
        self.symmetric_difference = GeoDataFrame(geometry=[Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)]), Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)])])
        self.a_identity_b = GeoDataFrame(geometry=[box(4, 2, 6, 4), Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)])])
        self.b_identity_a = GeoDataFrame(geometry=[box(4, 2, 6, 4), Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)])])

    def test_intersection(self):
        df_result = overlay(self.layer_a, self.layer_b, how='intersection')
        assert df_result.geom_equals(self.intersection).bool()

    def test_union(self):
        df_result = overlay(self.layer_a, self.layer_b, how='union')
        assert_geodataframe_equal(df_result, self.union)

    def test_a_difference_b(self):
        df_result = overlay(self.layer_a, self.layer_b, how='difference')
        assert_geodataframe_equal(df_result, self.a_difference_b)

    def test_b_difference_a(self):
        df_result = overlay(self.layer_b, self.layer_a, how='difference')
        assert_geodataframe_equal(df_result, self.b_difference_a)

    def test_symmetric_difference(self):
        df_result = overlay(self.layer_a, self.layer_b, how='symmetric_difference')
        assert_geodataframe_equal(df_result, self.symmetric_difference)

    def test_a_identity_b(self):
        df_result = overlay(self.layer_a, self.layer_b, how='identity')
        assert_geodataframe_equal(df_result, self.a_identity_b)

    def test_b_identity_a(self):
        df_result = overlay(self.layer_b, self.layer_a, how='identity')
        assert_geodataframe_equal(df_result, self.b_identity_a)