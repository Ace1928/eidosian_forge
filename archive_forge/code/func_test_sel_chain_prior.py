import importlib
import os
from collections import namedtuple
from copy import deepcopy
from html import escape
from typing import Dict
from tempfile import TemporaryDirectory
from urllib.parse import urlunsplit
import numpy as np
import pytest
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.testing import assert_identical
from ... import (
from ...data.base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs
from ...data.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata
from ..helpers import (  # pylint: disable=unused-import
def test_sel_chain_prior(self):
    idata = load_arviz_data('centered_eight')
    original_groups = getattr(idata, '_groups')
    idata_subset = idata.sel(inplace=False, chain_prior=False, chain=[0, 1, 3])
    groups = getattr(idata_subset, '_groups')
    assert np.all(np.isin(groups, original_groups))
    for group in groups:
        dataset_subset = getattr(idata_subset, group)
        dataset = getattr(idata, group)
        if 'chain' in dataset.dims:
            assert 'chain' in dataset_subset.dims
            if 'prior' not in group:
                assert np.all(dataset_subset.chain.values == np.array([0, 1, 3]))
        else:
            assert 'chain' not in dataset_subset.dims
    with pytest.raises(KeyError):
        idata.sel(inplace=False, chain_prior=True, chain=[0, 1, 3])