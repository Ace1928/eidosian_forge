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
def test_to_dataframe_bad(self):
    idata = from_dict(posterior={'a': np.random.randn(4, 100, 3, 4, 5), 'b': np.random.randn(4, 100)}, sample_stats={'a': np.random.randn(4, 100, 3, 4, 5), 'b': np.random.randn(4, 100)}, observed_data={'a': np.random.randn(3, 4, 5), 'b': np.random.randn(4)})
    with pytest.raises(TypeError):
        idata.to_dataframe(index_origin=2)
    with pytest.raises(TypeError):
        idata.to_dataframe(include_coords=False, include_index=False)
    with pytest.raises(TypeError):
        idata.to_dataframe(groups=['observed_data'])
    with pytest.raises(KeyError):
        idata.to_dataframe(groups=['invalid_group'])
    with pytest.raises(ValueError):
        idata.to_dataframe(var_names=['c'])