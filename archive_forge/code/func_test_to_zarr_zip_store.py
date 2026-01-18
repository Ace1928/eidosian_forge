from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
def test_to_zarr_zip_store(self, tmpdir, simple_datatree):
    from zarr.storage import ZipStore
    filepath = tmpdir / 'test.zarr.zip'
    original_dt = simple_datatree
    store = ZipStore(filepath)
    original_dt.to_zarr(store)
    roundtrip_dt = open_datatree(store, engine='zarr')
    assert_equal(original_dt, roundtrip_dt)