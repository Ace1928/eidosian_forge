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
def _arrow_to_geopandas(table, metadata=None):
    """
    Helper function with main, shared logic for read_parquet/read_feather.
    """
    df = table.to_pandas()
    metadata = metadata or table.schema.metadata
    if metadata is None or b'geo' not in metadata:
        raise ValueError('Missing geo metadata in Parquet/Feather file.\n            Use pandas.read_parquet/read_feather() instead.')
    try:
        metadata = _decode_metadata(metadata.get(b'geo', b''))
    except (TypeError, json.decoder.JSONDecodeError):
        raise ValueError('Missing or malformed geo metadata in Parquet/Feather file')
    _validate_metadata(metadata)
    geometry_columns = df.columns.intersection(metadata['columns'])
    if not len(geometry_columns):
        raise ValueError('No geometry columns are included in the columns read from\n            the Parquet/Feather file.  To read this file without geometry columns,\n            use pandas.read_parquet/read_feather() instead.')
    geometry = metadata['primary_column']
    if len(geometry_columns) and geometry not in geometry_columns:
        geometry = geometry_columns[0]
        if len(geometry_columns) > 1:
            warnings.warn('Multiple non-primary geometry columns read from Parquet/Feather file. The first column read was promoted to the primary geometry.', stacklevel=3)
    for col in geometry_columns:
        col_metadata = metadata['columns'][col]
        if 'crs' in col_metadata:
            crs = col_metadata['crs']
            if isinstance(crs, dict):
                _remove_id_from_member_of_ensembles(crs)
        else:
            crs = 'OGC:CRS84'
        df[col] = from_wkb(df[col].values, crs=crs)
    return GeoDataFrame(df, geometry=geometry)