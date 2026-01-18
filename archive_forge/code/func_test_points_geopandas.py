import dask.dataframe as dd
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray
import numpy as np
from numpy import nan
import pytest
@pytest.mark.skipif(not geodatasets, reason='geodatasets not installed')
@pytest.mark.skipif(not geopandas, reason='geopandas not installed')
@pytest.mark.parametrize('geom_type', ['multipoint', 'point'])
def test_points_geopandas(geom_type):
    df = geopandas.read_file(geodatasets.get_path('nybb'))
    df['geometry'] = df['geometry'].sample_points(100, rng=93814)
    if geom_type == 'point':
        df = df.explode(index_parts=False)
    unique_geom_type = df['geometry'].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type
    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.points(source=df, geometry='geometry', agg=ds.count())
    assert_eq_ndarray(agg.data, nybb_points_sol)