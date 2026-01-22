from unittest import SkipTest
import numpy as np
import pandas as pd
from shapely import geometry as sgeom
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path, Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import GeomTests
from geoviews.data import GeoPandasInterface
from .test_multigeometry import GeomInterfaceTest
class GeoPandasInterfaceTest(GeomInterfaceTest, GeomTests, RoundTripTests):
    """
    Test of the GeoPandasInterface.
    """
    datatype = 'geodataframe'
    interface = GeoPandasInterface
    __test__ = True

    def setUp(self):
        if geopandas is None:
            raise SkipTest('GeoPandasInterface requires geopandas, skipping tests')
        super().setUp()

    def test_df_dataset(self):
        if not pd:
            raise SkipTest('Pandas not available')
        dfs = [pd.DataFrame(np.column_stack([np.arange(i, i + 2), np.arange(i, i + 2)]), columns=['x', 'y']) for i in range(2)]
        mds = Path(dfs, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        for i, ds in enumerate(mds.split(datatype='dataframe')):
            ds['x'] = ds.x.astype(int)
            ds['y'] = ds.y.astype(int)
            self.assertEqual(ds, dfs[i])

    def test_multi_geom_point_coord_values(self):
        geoms = [{'geometry': sgeom.Point([(0, 1)])}, {'geometry': sgeom.Point([(3, 5)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(mds.dimension_values('x'), np.array([0, 3]))
        self.assertEqual(mds.dimension_values('y'), np.array([1, 5]))

    def test_multi_geom_point_length(self):
        geoms = [{'geometry': sgeom.Point([(0, 0)])}, {'geometry': sgeom.Point([(3, 3)])}]
        mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertEqual(len(mds), 2)

    def test_array_points_iloc_index_rows_index_cols(self):
        arrays = [np.array([(1 + i, i), (2 + i, i), (3 + i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        with self.assertRaises(DataError):
            mds.iloc[3, 0]

    def test_polygon_dtype(self):
        poly = Polygons([{'x': [1, 2, 3], 'y': [2, 0, 7]}], datatype=[self.datatype])
        self.assertIs(poly.interface, self.interface)
        self.assertEqual(poly.interface.dtype(poly, 'x'), 'float64')

    def test_geometry_column_not_named_geometry(self):
        gdf = geopandas.GeoDataFrame({'v': [1, 2], 'not geometry': [sgeom.Point(0, 1), sgeom.Point(1, 2)]}, geometry='not geometry')
        ds = Dataset(gdf, kdims=['Longitude', 'Latitude'], datatype=[self.datatype])
        self.assertEqual(ds.dimension_values('Longitude'), np.array([0, 1]))
        self.assertEqual(ds.dimension_values('Latitude'), np.array([1, 2]))

    def test_geometry_column_not_named_geometry_and_additional_geometry_column(self):
        gdf = geopandas.GeoDataFrame({'v': [1, 2], 'not geometry': [sgeom.Point(0, 1), sgeom.Point(1, 2)]}, geometry='not geometry')
        gdf = gdf.rename(columns={'v': 'geometry'})
        ds = Dataset(gdf, kdims=['Longitude', 'Latitude'], datatype=[self.datatype])
        self.assertEqual(ds.dimension_values('Longitude'), np.array([0, 1]))
        self.assertEqual(ds.dimension_values('Latitude'), np.array([1, 2]))