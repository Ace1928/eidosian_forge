from __future__ import annotations
import pickle
import pytest
import xarray as xr
from xarray.tests import assert_identical
def test_pickle_dataset(self) -> None:
    ds = xr.Dataset()
    ds_restored = pickle.loads(pickle.dumps(ds))
    assert_identical(ds, ds_restored)
    assert ds.example_accessor is ds.example_accessor
    ds.example_accessor.value = 'foo'
    ds_restored = pickle.loads(pickle.dumps(ds))
    assert_identical(ds, ds_restored)
    assert ds_restored.example_accessor.value == 'foo'