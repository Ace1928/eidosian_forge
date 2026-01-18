import datetime
import io
import os
import pathlib
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_series_equal
from shapely.geometry import Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.io.file import _detect_driver, _EXTENSION_TO_DRIVER
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
def test_to_file_schema(tmpdir, df_nybb, engine):
    """
    Ensure that the file is written according to the schema
    if it is specified

    """
    tempfilename = os.path.join(str(tmpdir), 'test.shp')
    properties = OrderedDict([('Shape_Leng', 'float:19.11'), ('BoroName', 'str:40'), ('BoroCode', 'int:10'), ('Shape_Area', 'float:19.11')])
    schema = {'geometry': 'Polygon', 'properties': properties}
    if engine == 'pyogrio':
        with pytest.raises(ValueError):
            df_nybb.iloc[:2].to_file(tempfilename, schema=schema, engine=engine)
    else:
        df_nybb.iloc[:2].to_file(tempfilename, schema=schema, engine=engine)
        import fiona
        with fiona.open(tempfilename) as f:
            result_schema = f.schema
        assert result_schema == schema