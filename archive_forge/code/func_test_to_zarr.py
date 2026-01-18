from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
def test_to_zarr(self, tmpdir, simple_datatree):
    filepath = tmpdir / 'test.zarr'
    original_dt = simple_datatree
    original_dt.to_zarr(filepath)
    roundtrip_dt = open_datatree(filepath, engine='zarr')
    assert_equal(original_dt, roundtrip_dt)