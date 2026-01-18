from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
dask.graph_manipulation passes an optional parameter, "rename", to the rebuilder
    function returned by __dask_postperist__; also, the dsk passed to the rebuilder is
    a HighLevelGraph whereas with dask.persist() and dask.optimize() it's a plain dict.
    