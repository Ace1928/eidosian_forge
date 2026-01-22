import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
class TestGeometryArrayCRS:

    def setup_method(self):
        self.osgb = pyproj.CRS(27700)
        self.wgs = pyproj.CRS(4326)
        self.geoms = [Point(0, 0), Point(1, 1)]
        self.polys = [Polygon([(random.random(), random.random()) for i in range(3)]) for _ in range(10)]
        self.arr = from_shapely(self.polys, crs=27700)

    def test_array(self):
        arr = from_shapely(self.geoms)
        arr.crs = 27700
        assert arr.crs == self.osgb
        arr = from_shapely(self.geoms, crs=27700)
        assert arr.crs == self.osgb
        arr = GeometryArray(arr)
        assert arr.crs == self.osgb
        arr = GeometryArray(arr, crs=4326)
        assert arr.crs == self.wgs

    def test_series(self):
        s = GeoSeries(crs=27700)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb
        s.crs = 4326
        assert s.crs == self.wgs
        assert s.values.crs == self.wgs
        s = GeoSeries(self.geoms, crs=27700)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(arr)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb
        with pytest.raises(ValueError, match="CRS mismatch between CRS of the passed geometries and 'crs'"):
            s = GeoSeries(arr, crs=4326)
        assert s.crs == self.osgb

    def test_dataframe(self):
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame(geometry=arr)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        df = GeoDataFrame(geometry=s)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        match_str = "CRS mismatch between CRS of the passed geometries and 'crs'"
        with pytest.raises(ValueError, match=match_str):
            df = GeoDataFrame(geometry=s, crs=4326)
        with pytest.raises(ValueError, match=match_str):
            GeoDataFrame(geometry=s, crs=4326)
        with pytest.raises(ValueError, match=match_str):
            GeoDataFrame({'data': [1, 2], 'geometry': s}, crs=4326)
        with pytest.raises(ValueError, match=match_str):
            GeoDataFrame(df, crs=4326).crs
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        df = GeoDataFrame(geometry=s)
        df.crs = 4326
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs
        with pytest.raises(ValueError, match='Assigning CRS to a GeoDataFrame without'):
            GeoDataFrame(self.geoms, columns=['geom'], crs=27700)
        with pytest.raises(ValueError, match='Assigning CRS to a GeoDataFrame without'):
            GeoDataFrame(crs=27700)
        df = GeoDataFrame(self.geoms, columns=['geom'])
        df = df.set_geometry('geom', crs=27700)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.geom.crs == self.osgb
        assert df.geom.values.crs == self.osgb
        df = GeoDataFrame(geometry=self.geoms, crs=27700)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        df = GeoDataFrame(geometry=self.geoms, crs=27700)
        df = df.set_geometry(self.geoms, crs=4326)
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        df = GeoDataFrame()
        df = df.set_geometry(s)
        assert df._geometry_column_name == 'geometry'
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame()
        df = df.set_geometry(arr)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms)
        df = GeoDataFrame({'col1': [1, 2], 'geometry': arr}, crs=4326)
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs
        arr = from_shapely(self.geoms, crs=4326)
        df = GeoDataFrame({'col1': [1, 2], 'geometry': arr})
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs
        df = GeoDataFrame({'geometry': [0, 1]})
        with pytest.raises(ValueError, match='Assigning CRS to a GeoDataFrame without a geometry'):
            df.crs = 27700
        df = GeoDataFrame({'geometry': [Point(0, 1)]}).assign(geometry=[0])
        with pytest.raises(ValueError, match='Assigning CRS to a GeoDataFrame without an active geometry'):
            df.crs = 27700
        with pytest.raises(AttributeError, match='The CRS attribute of a GeoDataFrame without an active'):
            assert df.crs == self.osgb

    def test_dataframe_getitem_without_geometry_column(self):
        df = GeoDataFrame({'col': range(10)}, geometry=self.arr)
        df['geom2'] = df.geometry.centroid
        subset = df[['col', 'geom2']]
        with pytest.raises(AttributeError, match='The CRS attribute of a GeoDataFrame without an active'):
            assert subset.crs == self.osgb

    def test_dataframe_setitem(self):
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        df = GeoDataFrame()
        with pytest.warns(FutureWarning, match="You are adding a column named 'geometry'"):
            df['geometry'] = s
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame()
        with pytest.warns(FutureWarning, match="You are adding a column named 'geometry'"):
            df['geometry'] = arr
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms)
        df = GeoDataFrame({'col1': [1, 2], 'geometry': arr}, crs=4326)
        df['geometry'] = df['geometry'].to_crs(27700)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms)
        df = GeoDataFrame({'col1': [1, 2], 'geometry': arr, 'other_geom': arr}, crs=4326)
        df['other_geom'] = from_shapely(self.geoms, crs=27700)
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df['geometry'].crs == self.wgs
        assert df['other_geom'].crs == self.osgb

    def test_dataframe_setitem_without_geometry_column(self):
        arr = from_shapely(self.geoms)
        df = GeoDataFrame({'col1': [1, 2], 'geometry': arr}, crs=4326)
        with pytest.warns(UserWarning):
            df['geometry'] = 1
        df['geometry'] = self.geoms
        assert df.crs is None

    @pytest.mark.parametrize('scalar', [None, Point(0, 0), LineString([(0, 0), (1, 1)])])
    def test_scalar(self, scalar):
        df = GeoDataFrame()
        with pytest.warns(FutureWarning, match="You are adding a column named 'geometry'"):
            df['geometry'] = scalar
        df.crs = 4326
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs

    @pytest.mark.filterwarnings('ignore:Accessing CRS')
    def test_crs_with_no_geom_fails(self):
        with pytest.raises(ValueError, match='Assigning CRS to a GeoDataFrame without'):
            df = GeoDataFrame()
            df.crs = 4326

    def test_read_file(self):
        nybb_filename = datasets.get_path('nybb')
        df = read_file(nybb_filename)
        assert df.crs == pyproj.CRS(2263)
        assert df.geometry.crs == pyproj.CRS(2263)
        assert df.geometry.values.crs == pyproj.CRS(2263)

    def test_multiple_geoms(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=['col1'])
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.col1.crs == self.wgs
        assert df.col1.values.crs == self.wgs

    def test_multiple_geoms_set_geom(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=['col1'])
        df = df.set_geometry('col1')
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs
        assert df['geometry'].crs == self.osgb
        assert df['geometry'].values.crs == self.osgb

    def test_assign_cols(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=['col1'])
        df['geom2'] = s
        df['geom3'] = s.values
        df['geom4'] = from_shapely(self.geoms)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.geom2.crs == self.wgs
        assert df.geom2.values.crs == self.wgs
        assert df.geom3.crs == self.wgs
        assert df.geom3.values.crs == self.wgs
        assert df.geom4.crs is None
        assert df.geom4.values.crs is None

    def test_copy(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=['col1'])
        arr_copy = arr.copy()
        assert arr_copy.crs == arr.crs
        s_copy = s.copy()
        assert s_copy.crs == s.crs
        assert s_copy.values.crs == s.values.crs
        df_copy = df.copy()
        assert df_copy.crs == df.crs
        assert df_copy.geometry.crs == df.geometry.crs
        assert df_copy.geometry.values.crs == df.geometry.values.crs
        assert df_copy.col1.crs == df.col1.crs
        assert df_copy.col1.values.crs == df.col1.values.crs

    def test_rename(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=['col1'])
        df = df.rename(columns={'geometry': 'geom'}).set_geometry('geom')
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        df = df.rename_geometry('geom2')
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        df = df.rename(columns={'col1': 'column1'})
        assert df.column1.crs == self.wgs
        assert df.column1.values.crs == self.wgs

    def test_geoseries_to_crs(self):
        s = GeoSeries(self.geoms, crs=27700)
        s = s.to_crs(4326)
        assert s.crs == self.wgs
        assert s.values.crs == self.wgs
        df = GeoDataFrame(geometry=s)
        assert df.crs == self.wgs
        df = df.to_crs(27700)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        arr = from_shapely(self.geoms, crs=4326)
        df['col1'] = arr
        df = df.to_crs(3857)
        assert df.col1.crs == self.wgs
        assert df.col1.values.crs == self.wgs

    def test_array_to_crs(self):
        arr = from_shapely(self.geoms, crs=27700)
        arr = arr.to_crs(4326)
        assert arr.crs == self.wgs

    def test_from_shapely(self):
        arr = from_shapely(self.geoms, crs=27700)
        assert arr.crs == self.osgb

    def test_from_wkb(self):
        L_wkb = [p.wkb for p in self.geoms]
        arr = from_wkb(L_wkb, crs=27700)
        assert arr.crs == self.osgb

    def test_from_wkt(self):
        L_wkt = [p.wkt for p in self.geoms]
        arr = from_wkt(L_wkt, crs=27700)
        assert arr.crs == self.osgb

    def test_points_from_xy(self):
        df = pd.DataFrame([{'x': x, 'y': x, 'z': x} for x in range(10)])
        arr = points_from_xy(df['x'], df['y'], crs=27700)
        assert arr.crs == self.osgb

    def test_original(self):
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        assert arr.crs is None
        assert s.crs == self.osgb

    def test_ops(self):
        arr = self.arr
        bound = arr.boundary
        assert bound.crs == self.osgb
        cent = arr.centroid
        assert cent.crs == self.osgb
        hull = arr.convex_hull
        assert hull.crs == self.osgb
        envelope = arr.envelope
        assert envelope.crs == self.osgb
        exterior = arr.exterior
        assert exterior.crs == self.osgb
        representative_point = arr.representative_point()
        assert representative_point.crs == self.osgb

    def test_binary_ops(self):
        arr = self.arr
        quads = []
        while len(quads) < 10:
            geom = Polygon([(random.random(), random.random()) for i in range(4)])
            if geom.is_valid:
                quads.append(geom)
        arr2 = from_shapely(quads, crs=27700)
        difference = arr.difference(arr2)
        assert difference.crs == self.osgb
        intersection = arr.intersection(arr2)
        assert intersection.crs == self.osgb
        symmetric_difference = arr.symmetric_difference(arr2)
        assert symmetric_difference.crs == self.osgb
        union = arr.union(arr2)
        assert union.crs == self.osgb

    def test_other(self):
        arr = self.arr
        buffer = arr.buffer(5)
        assert buffer.crs == self.osgb
        interpolate = arr.exterior.interpolate(0.1)
        assert interpolate.crs == self.osgb
        simplify = arr.simplify(5)
        assert simplify.crs == self.osgb

    @pytest.mark.parametrize('attr, arg', [('affine_transform', ([0, 1, 1, 0, 0, 0],)), ('translate', ()), ('rotate', (10,)), ('scale', ()), ('skew', ())])
    def test_affinity_methods(self, attr, arg):
        result = getattr(self.arr, attr)(*arg)
        assert result.crs == self.osgb

    def test_slice(self):
        s = GeoSeries(self.arr, crs=27700)
        assert s.iloc[1:].values.crs == self.osgb
        df = GeoDataFrame({'col1': self.arr}, geometry=s)
        assert df.iloc[1:].geometry.values.crs == self.osgb
        assert df.iloc[1:].col1.values.crs == self.osgb

    def test_concat(self):
        s = GeoSeries(self.arr, crs=27700)
        assert pd.concat([s, s]).values.crs == self.osgb
        df = GeoDataFrame({'col1': from_shapely(self.geoms, crs=4326)}, geometry=s)
        assert pd.concat([df, df]).geometry.values.crs == self.osgb
        assert pd.concat([df, df]).col1.values.crs == self.wgs

    def test_merge(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame({'col1': s}, geometry=arr)
        df2 = GeoDataFrame({'col2': s}, geometry=arr).rename_geometry('geom')
        merged = df.merge(df2, left_index=True, right_index=True)
        assert merged.col1.values.crs == self.wgs
        assert merged.geometry.values.crs == self.osgb
        assert merged.col2.values.crs == self.wgs
        assert merged.geom.values.crs == self.osgb
        assert merged.crs == self.osgb

    def test_setitem_geometry(self):
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame({'col1': [0, 1]}, geometry=arr)
        df['geometry'] = list(df.geometry)
        assert df.geometry.values.crs == self.osgb
        df2 = GeoDataFrame({'col1': [0, 1]}, geometry=arr)
        df2['geometry'] = from_shapely(self.geoms, crs=4326)
        assert df2.geometry.values.crs == self.wgs

    def test_astype(self):
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame({'col1': [0, 1]}, geometry=arr)
        df2 = df.astype({'col1': str})
        assert df2.crs == self.osgb

    def test_apply(self):
        s = GeoSeries(self.arr)
        assert s.crs == 27700
        result = s.apply(lambda x: x.centroid)
        assert result.crs == 27700

    def test_apply_geodataframe(self):
        df = GeoDataFrame({'col1': [0, 1]}, geometry=self.geoms, crs=27700)
        assert df.crs == 27700
        result = df.apply(lambda col: col, axis=0)
        assert result.crs == 27700
        result = df.apply(lambda row: row, axis=1)
        assert result.crs == 27700