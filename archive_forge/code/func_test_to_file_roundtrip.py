import os
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
import pytest
from .test_file import FIONA_MARK, PYOGRIO_MARK
def test_to_file_roundtrip(tmpdir, geodataframe, ogr_driver, engine):
    output_file = os.path.join(str(tmpdir), 'output_file')
    write_kwargs = {}
    if ogr_driver == 'SQLite':
        write_kwargs['spatialite'] = True
        if engine == 'fiona':
            import fiona
            from packaging.version import Version
            if Version(fiona.__version__) < Version('1.8.20'):
                pytest.skip('SQLite driver only available from version 1.8.20')
        if engine == 'pyogrio' and len(geodataframe == 2) and (geodataframe.geometry[0] is None) and (geodataframe.geometry[1] is not None) and geodataframe.geometry[1].has_z:
            write_kwargs['geometry_type'] = 'Point Z'
    expected_error = _expected_error_on(geodataframe, ogr_driver)
    if expected_error:
        with pytest.raises(RuntimeError, match='Failed to write record|Could not add feature to layer'):
            geodataframe.to_file(output_file, driver=ogr_driver, engine=engine, **write_kwargs)
    else:
        geodataframe.to_file(output_file, driver=ogr_driver, engine=engine, **write_kwargs)
        reloaded = geopandas.read_file(output_file, engine=engine)
        if ogr_driver == 'GeoJSON' and engine == 'pyogrio':
            reloaded['a'] = reloaded['a'].astype('int64')
        assert_geodataframe_equal(geodataframe, reloaded, check_column_type='equiv')