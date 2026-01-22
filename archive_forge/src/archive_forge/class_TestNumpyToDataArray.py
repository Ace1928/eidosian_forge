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
class TestNumpyToDataArray:

    def test_1d_dataset(self):
        size = 100
        dataset = convert_to_dataset(np.random.randn(size))
        assert len(dataset.data_vars) == 1
        assert set(dataset.coords) == {'chain', 'draw'}
        assert dataset.chain.shape == (1,)
        assert dataset.draw.shape == (size,)

    def test_warns_bad_shape(self):
        with pytest.warns(UserWarning):
            convert_to_dataset(np.random.randn(100, 4))

    def test_nd_to_dataset(self):
        shape = (1, 2, 3, 4, 5)
        dataset = convert_to_dataset(np.random.randn(*shape))
        assert len(dataset.data_vars) == 1
        var_name = list(dataset.data_vars)[0]
        assert len(dataset.coords) == len(shape)
        assert dataset.chain.shape == shape[:1]
        assert dataset.draw.shape == shape[1:2]
        assert dataset[var_name].shape == shape

    def test_nd_to_inference_data(self):
        shape = (1, 2, 3, 4, 5)
        inference_data = convert_to_inference_data(np.random.randn(*shape), group='prior')
        assert hasattr(inference_data, 'prior')
        assert len(inference_data.prior.data_vars) == 1
        var_name = list(inference_data.prior.data_vars)[0]
        assert len(inference_data.prior.coords) == len(shape)
        assert inference_data.prior.chain.shape == shape[:1]
        assert inference_data.prior.draw.shape == shape[1:2]
        assert inference_data.prior[var_name].shape == shape
        assert repr(inference_data).startswith('Inference data with groups')

    def test_more_chains_than_draws(self):
        shape = (10, 4)
        with pytest.warns(UserWarning):
            inference_data = convert_to_inference_data(np.random.randn(*shape), group='prior')
        assert hasattr(inference_data, 'prior')
        assert len(inference_data.prior.data_vars) == 1
        var_name = list(inference_data.prior.data_vars)[0]
        assert len(inference_data.prior.coords) == len(shape)
        assert inference_data.prior.chain.shape == shape[:1]
        assert inference_data.prior.draw.shape == shape[1:2]
        assert inference_data.prior[var_name].shape == shape