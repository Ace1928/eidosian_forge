import os
from packaging.version import Version
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import pyproj
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from geopandas import GeoDataFrame, GeoSeries
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc, uses_params, uses_relative
import urllib.request
def _read_file_fiona(path_or_bytes, from_bytes, bbox=None, mask=None, rows=None, where=None, **kwargs):
    if where is not None and (not FIONA_GE_19):
        raise NotImplementedError('where requires fiona 1.9+')
    if not from_bytes:
        if _is_zip(str(path_or_bytes)):
            parsed = fiona.parse_path(str(path_or_bytes))
            if isinstance(parsed, fiona.path.ParsedPath):
                schemes = (parsed.scheme or '').split('+')
                if 'zip' not in schemes:
                    parsed.scheme = '+'.join(['zip'] + schemes)
                path_or_bytes = parsed.name
            elif isinstance(parsed, fiona.path.UnparsedPath) and (not str(path_or_bytes).startswith('/vsi')):
                path_or_bytes = 'zip://' + parsed.name
    if from_bytes:
        reader = fiona.BytesCollection
    else:
        reader = fiona.open
    with fiona_env():
        with reader(path_or_bytes, **kwargs) as features:
            crs = features.crs_wkt
            try:
                epsg = features.crs.to_epsg(confidence_threshold=100)
                if epsg is not None:
                    crs = epsg
            except AttributeError:
                try:
                    crs = features.crs['init']
                except (TypeError, KeyError):
                    pass
            if bbox is not None:
                if isinstance(bbox, (GeoDataFrame, GeoSeries)):
                    bbox = tuple(bbox.to_crs(crs).total_bounds)
                elif isinstance(bbox, BaseGeometry):
                    bbox = bbox.bounds
                assert len(bbox) == 4
            elif isinstance(mask, (GeoDataFrame, GeoSeries)):
                mask = mapping(mask.to_crs(crs).unary_union)
            elif isinstance(mask, BaseGeometry):
                mask = mapping(mask)
            filters = {}
            if bbox is not None:
                filters['bbox'] = bbox
            if mask is not None:
                filters['mask'] = mask
            if where is not None:
                filters['where'] = where
            if rows is not None:
                if isinstance(rows, int):
                    rows = slice(rows)
                elif not isinstance(rows, slice):
                    raise TypeError("'rows' must be an integer or a slice.")
                f_filt = features.filter(rows.start, rows.stop, rows.step, **filters)
            elif filters:
                f_filt = features.filter(**filters)
            else:
                f_filt = features
            columns = list(features.schema['properties'])
            datetime_fields = [k for k, v in features.schema['properties'].items() if v == 'datetime']
            if kwargs.get('ignore_geometry', False):
                df = pd.DataFrame([record['properties'] for record in f_filt], columns=columns)
            else:
                df = GeoDataFrame.from_features(f_filt, crs=crs, columns=columns + ['geometry'])
            for k in datetime_fields:
                as_dt = None
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error', 'In a future version of pandas, parsing datetimes with mixed time zones will raise an error', FutureWarning)
                        as_dt = pd.to_datetime(df[k])
                except Exception:
                    pass
                if as_dt is None or as_dt.dtype == 'object':
                    try:
                        as_dt = pd.to_datetime(df[k], utc=True)
                    except Exception:
                        pass
                if as_dt is not None and (not as_dt.dtype == 'object'):
                    df[k] = as_dt.dt.round(freq='ms')
            return df