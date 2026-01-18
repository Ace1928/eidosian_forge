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
def test_to_dict_warmup(self):
    idata = create_data_random(groups=['posterior', 'sample_stats', 'observed_data', 'warmup_posterior', 'warmup_posterior_predictive'])
    test_data = from_dict(**idata.to_dict(), save_warmup=True)
    assert test_data
    for group in idata._groups_all:
        xr_data = getattr(idata, group)
        test_xr_data = getattr(test_data, group)
        assert xr_data.equals(test_xr_data)