from contextlib import ExitStack
import logging
import os
import warnings
from collections import OrderedDict
from fiona import compat, vfs
from fiona.ogrext import Iterator, ItemsIterator, KeysIterator
from fiona.ogrext import Session, WritingSession
from fiona.ogrext import buffer_to_virtual_file, remove_virtual_file, GEOMETRY_TYPES
from fiona.errors import (
from fiona.logutils import FieldSkipLogFilter
from fiona.crs import CRS
from fiona._env import get_gdal_release_name, get_gdal_version_tuple
from fiona.env import env_ctx_if_needed
from fiona.errors import FionaDeprecationWarning
from fiona.drvsupport import (
from fiona.path import Path, vsi_path, parse_path
def _get_valid_geom_types(schema, driver):
    """Returns a set of geometry types the schema will accept"""
    schema_geom_type = schema['geometry']
    if isinstance(schema_geom_type, str) or schema_geom_type is None:
        schema_geom_type = (schema_geom_type,)
    valid_types = set()
    for geom_type in schema_geom_type:
        geom_type = str(geom_type).lstrip('3D ')
        if geom_type == 'Unknown' or geom_type == 'Any':
            valid_types.update(ALL_GEOMETRY_TYPES)
        else:
            if geom_type not in ALL_GEOMETRY_TYPES:
                raise UnsupportedGeometryTypeError(geom_type)
            valid_types.add(geom_type)
    if driver == 'ESRI Shapefile' and 'Point' not in valid_types:
        for geom_type in list(valid_types):
            if not geom_type.startswith('Multi'):
                valid_types.add('Multi' + geom_type)
    return valid_types