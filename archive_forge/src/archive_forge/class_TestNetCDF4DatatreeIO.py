from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
@requires_netCDF4
class TestNetCDF4DatatreeIO(DatatreeIOBase):
    engine: T_NetcdfEngine | None = 'netcdf4'