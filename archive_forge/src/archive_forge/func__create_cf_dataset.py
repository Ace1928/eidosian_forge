from __future__ import annotations
import contextlib
import gzip
import itertools
import math
import os.path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from io import BytesIO
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.errors import OutOfBoundsDatetime
import xarray as xr
from xarray import (
from xarray.backends.common import robust_getitem
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.backends.netCDF4_ import (
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.scipy_ import ScipyBackendEntrypoint
from xarray.coding.cftime_offsets import cftime_range
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_coding_times import (
from xarray.tests.test_dataset import (
@staticmethod
def _create_cf_dataset():
    original = Dataset(dict(variable=(('ln_p', 'latitude', 'longitude'), np.arange(8, dtype='f4').reshape(2, 2, 2), {'ancillary_variables': 'std_devs det_lim'}), std_devs=(('ln_p', 'latitude', 'longitude'), np.arange(0.1, 0.9, 0.1).reshape(2, 2, 2), {'standard_name': 'standard_error'}), det_lim=((), 0.1, {'standard_name': 'detection_minimum'})), dict(latitude=('latitude', [0, 1], {'units': 'degrees_north'}), longitude=('longitude', [0, 1], {'units': 'degrees_east'}), latlon=((), -1, {'grid_mapping_name': 'latitude_longitude'}), latitude_bnds=(('latitude', 'bnds2'), [[0, 1], [1, 2]]), longitude_bnds=(('longitude', 'bnds2'), [[0, 1], [1, 2]]), areas=(('latitude', 'longitude'), [[1, 1], [1, 1]], {'units': 'degree^2'}), ln_p=('ln_p', [1.0, 0.5], {'standard_name': 'atmosphere_ln_pressure_coordinate', 'computed_standard_name': 'air_pressure'}), P0=((), 1013.25, {'units': 'hPa'})))
    original['variable'].encoding.update({'cell_measures': 'area: areas', 'grid_mapping': 'latlon'})
    original.coords['latitude'].encoding.update(dict(grid_mapping='latlon', bounds='latitude_bnds'))
    original.coords['longitude'].encoding.update(dict(grid_mapping='latlon', bounds='longitude_bnds'))
    original.coords['ln_p'].encoding.update({'formula_terms': 'p0: P0 lev : ln_p'})
    return original