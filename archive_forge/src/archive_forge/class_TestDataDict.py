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
class TestDataDict:

    @pytest.fixture(scope='class')
    def data(self, draws, chains):

        class Data:
            obj = {}
            for key, shape in {'mu': [], 'tau': [], 'eta': [8], 'theta': [8]}.items():
                obj[key] = np.random.randn(chains, draws, *shape)
        return Data

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {'mu', 'tau', 'eta', 'theta'}
        assert set(dataset.coords) == {'chain', 'draw', 'school'}

    def get_inference_data(self, data, eight_schools_params, save_warmup=False):
        return from_dict(posterior=data.obj, posterior_predictive=data.obj, sample_stats=data.obj, prior=data.obj, prior_predictive=data.obj, sample_stats_prior=data.obj, warmup_posterior=data.obj, warmup_posterior_predictive=data.obj, predictions=data.obj, observed_data=eight_schools_params, coords={'school': np.arange(8)}, pred_coords={'school_pred': np.arange(8)}, dims={'theta': ['school'], 'eta': ['school']}, pred_dims={'theta': ['school_pred'], 'eta': ['school_pred']}, save_warmup=save_warmup)

    def test_inference_data(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {'posterior': [], 'prior': [], 'sample_stats': [], 'posterior_predictive': [], 'prior_predictive': [], 'sample_stats_prior': [], 'observed_data': ['J', 'y', 'sigma']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        self.check_var_names_coords_dims(inference_data.posterior)
        self.check_var_names_coords_dims(inference_data.posterior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats)
        self.check_var_names_coords_dims(inference_data.prior)
        self.check_var_names_coords_dims(inference_data.prior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats_prior)
        pred_dims = inference_data.predictions.sizes['school_pred']
        assert pred_dims == 8

    def test_inference_data_warmup(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params, save_warmup=True)
        test_dict = {'posterior': [], 'prior': [], 'sample_stats': [], 'posterior_predictive': [], 'prior_predictive': [], 'sample_stats_prior': [], 'observed_data': ['J', 'y', 'sigma'], 'warmup_posterior_predictive': [], 'warmup_posterior': []}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        self.check_var_names_coords_dims(inference_data.posterior)
        self.check_var_names_coords_dims(inference_data.posterior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats)
        self.check_var_names_coords_dims(inference_data.prior)
        self.check_var_names_coords_dims(inference_data.prior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats_prior)
        self.check_var_names_coords_dims(inference_data.warmup_posterior)
        self.check_var_names_coords_dims(inference_data.warmup_posterior_predictive)

    def test_inference_data_edge_cases(self):
        log_likelihood = {'y': np.random.randn(4, 100), 'log_likelihood': np.random.randn(4, 100, 8)}
        with pytest.warns(UserWarning, match='log_likelihood.+in posterior'):
            assert from_dict(posterior=log_likelihood) is not None
        assert from_dict(observed_data=log_likelihood, dims=None) is not None

    def test_inference_data_bad(self):
        x = np.random.randn(4, 100)
        with pytest.raises(TypeError):
            from_dict(posterior=x)
        with pytest.raises(TypeError):
            from_dict(posterior_predictive=x)
        with pytest.raises(TypeError):
            from_dict(sample_stats=x)
        with pytest.raises(TypeError):
            from_dict(prior=x)
        with pytest.raises(TypeError):
            from_dict(prior_predictive=x)
        with pytest.raises(TypeError):
            from_dict(sample_stats_prior=x)
        with pytest.raises(TypeError):
            from_dict(observed_data=x)

    def test_from_dict_warning(self):
        bad_posterior_dict = {'log_likelihood': np.ones((5, 1000, 2))}
        with pytest.warns(UserWarning):
            from_dict(posterior=bad_posterior_dict)