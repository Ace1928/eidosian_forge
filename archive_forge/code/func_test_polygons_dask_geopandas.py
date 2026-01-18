import dask.dataframe as dd
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray
import numpy as np
from numpy import nan
import pytest
@pytest.mark.skipif(not geodatasets, reason='geodatasets not installed')
@pytest.mark.skipif(not dask_geopandas, reason='dask_geopandas not installed')
@pytest.mark.skipif(not geopandas, reason='geopandas not installed')
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize('geom_type', ['multipolygon', 'polygon'])
def test_polygons_dask_geopandas(geom_type, npartitions):
    df = geopandas.read_file(geodatasets.get_path('nybb'))
    df['col'] = np.arange(len(df))
    if geom_type == 'polygon':
        df = df.explode(index_parts=False)
    unique_geom_type = df['geometry'].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type
    df = dd.from_pandas(df, npartitions=npartitions)
    assert df.npartitions == npartitions
    df.calculate_spatial_partitions()
    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.polygons(source=df, geometry='geometry', agg=ds.max('col'))
    assert_eq_ndarray(agg.data, nybb_polygons_sol)