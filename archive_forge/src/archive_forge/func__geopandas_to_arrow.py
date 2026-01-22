from packaging.version import Version
import json
import warnings
import numpy as np
from pandas import DataFrame, Series
import geopandas._compat as compat
import shapely
from geopandas._compat import import_optional_dependency
from geopandas.array import from_wkb
from geopandas import GeoDataFrame
import geopandas
from .file import _expand_user
def _geopandas_to_arrow(df, index=None, schema_version=None):
    """
    Helper function with main, shared logic for to_parquet/to_feather.
    """
    from pyarrow import Table
    _validate_dataframe(df)
    geo_metadata = _create_metadata(df, schema_version=schema_version)
    kwargs = {}
    if compat.USE_SHAPELY_20 and shapely.geos.geos_version > (3, 10, 0):
        kwargs = {'flavor': 'iso'}
    else:
        for col in df.columns[df.dtypes == 'geometry']:
            series = df[col]
            if series.has_z.any():
                warnings.warn('The GeoDataFrame contains 3D geometries, and when using shapely < 2.0 or GEOS < 3.10, such geometries will be written not exactly following to the GeoParquet spec (not using ISO WKB). For most use cases this should not be a problem (GeoPandas can read such files fine).', stacklevel=2)
                break
    df = df.to_wkb(**kwargs)
    table = Table.from_pandas(df, preserve_index=index)
    metadata = table.schema.metadata
    metadata.update({b'geo': _encode_metadata(geo_metadata)})
    return table.replace_schema_metadata(metadata)