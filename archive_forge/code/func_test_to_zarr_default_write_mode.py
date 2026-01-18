from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
def test_to_zarr_default_write_mode(self, tmpdir, simple_datatree):
    import zarr
    simple_datatree.to_zarr(tmpdir)
    with pytest.raises(zarr.errors.ContainsGroupError):
        simple_datatree.to_zarr(tmpdir)