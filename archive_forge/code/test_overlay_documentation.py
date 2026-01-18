import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest

    Test of mixed geometry types on input and output. Expected results initially
    generated using following snippet.

        polys1 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        df1 = gpd.GeoDataFrame({'col1': [1, 2], 'geometry': polys1})

        polys2 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        df2 = gpd.GeoDataFrame({'geometry': polys2, 'col2': [1, 2, 3]})

        lines1 = gpd.GeoSeries([LineString([(2, 0), (2, 4), (6, 4)]),
                                LineString([(0, 3), (6, 3)])])
        df3 = gpd.GeoDataFrame({'col3': [1, 2], 'geometry': lines1})
        points1 = gpd.GeoSeries([Point((2, 2)),
                                 Point((3, 3))])
        df4 = gpd.GeoDataFrame({'col4': [1, 2], 'geometry': points1})

        params=["union", "intersection", "difference", "symmetric_difference",
                "identity"]
        stricts = [True, False]

        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df2, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('polys_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')

        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df3, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('poly_line_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')
        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df4, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('poly_point_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')
    