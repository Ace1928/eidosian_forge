import re
from collections import OrderedDict
from copy import deepcopy
from math import ceil
import numpy as np
import xarray as xr
from .. import _log
from ..rcparams import rcParams
from .base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
class PyStan3Converter:
    """Encapsulate PyStan3 specific logic."""

    def __init__(self, *, posterior=None, posterior_model=None, posterior_predictive=None, predictions=None, prior=None, prior_model=None, prior_predictive=None, observed_data=None, constant_data=None, predictions_constant_data=None, log_likelihood=None, coords=None, dims=None, save_warmup=None, dtypes=None):
        self.posterior = posterior
        self.posterior_model = posterior_model
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_model = prior_model
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = rcParams['data.log_likelihood'] if log_likelihood is None else log_likelihood
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams['data.save_warmup'] if save_warmup is None else save_warmup
        self.dtypes = dtypes
        if self.log_likelihood is True and self.posterior is not None and ('log_lik' in self.posterior.param_names):
            self.log_likelihood = ['log_lik']
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None
        import stan
        self.stan = stan

    @requires('posterior')
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [predictions]
        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        elif isinstance(log_likelihood, dict):
            log_likelihood = list(log_likelihood.values())
        ignore = posterior_predictive + predictions + log_likelihood
        data, data_warmup = get_draws_stan3(posterior, model=posterior_model, ignore=ignore, warmup=self.save_warmup, dtypes=self.dtypes)
        attrs = get_attrs_stan3(posterior, model=posterior_model)
        return (dict_to_dataset(data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims))

    @requires('posterior')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        data, data_warmup = get_sample_stats_stan3(posterior, ignore='lp__', warmup=self.save_warmup, dtypes=self.dtypes)
        data_lp, data_warmup_lp = get_sample_stats_stan3(posterior, variables='lp__', warmup=self.save_warmup)
        data['lp'] = data_lp['lp']
        if data_warmup_lp:
            data_warmup['lp'] = data_warmup_lp['lp']
        attrs = get_attrs_stan3(posterior, model=posterior_model)
        return (dict_to_dataset(data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims))

    @requires('posterior')
    @requires('log_likelihood')
    def log_likelihood_to_xarray(self):
        """Store log_likelihood data in log_likelihood group."""
        fit = self.posterior
        log_likelihood = self.log_likelihood
        model = self.posterior_model
        if isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        if isinstance(log_likelihood, (list, tuple)):
            log_likelihood = {name: name for name in log_likelihood}
        log_likelihood_draws, log_likelihood_draws_warmup = get_draws_stan3(fit, model=model, variables=list(log_likelihood.values()), warmup=self.save_warmup, dtypes=self.dtypes)
        data = {obs_var_name: log_likelihood_draws[log_like_name] for obs_var_name, log_like_name in log_likelihood.items() if log_like_name in log_likelihood_draws}
        data_warmup = {obs_var_name: log_likelihood_draws_warmup[log_like_name] for obs_var_name, log_like_name in log_likelihood.items() if log_like_name in log_likelihood_draws_warmup}
        return (dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims))

    @requires('posterior')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        posterior_predictive = self.posterior_predictive
        data, data_warmup = get_draws_stan3(posterior, model=posterior_model, variables=posterior_predictive, warmup=self.save_warmup, dtypes=self.dtypes)
        return (dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims))

    @requires('posterior')
    @requires('predictions')
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        predictions = self.predictions
        data, data_warmup = get_draws_stan3(posterior, model=posterior_model, variables=predictions, warmup=self.save_warmup, dtypes=self.dtypes)
        return (dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims))

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        prior_predictive = self.prior_predictive
        if prior_predictive is None:
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]
        ignore = prior_predictive
        data, data_warmup = get_draws_stan3(prior, model=prior_model, ignore=ignore, warmup=self.save_warmup, dtypes=self.dtypes)
        attrs = get_attrs_stan3(prior, model=prior_model)
        return (dict_to_dataset(data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims))

    @requires('prior')
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        prior_model = self.prior_model
        data, data_warmup = get_sample_stats_stan3(prior, warmup=self.save_warmup, dtypes=self.dtypes)
        attrs = get_attrs_stan3(prior, model=prior_model)
        return (dict_to_dataset(data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims))

    @requires('prior')
    @requires('prior_predictive')
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        prior_predictive = self.prior_predictive
        data, data_warmup = get_draws_stan3(prior, model=prior_model, variables=prior_predictive, warmup=self.save_warmup, dtypes=self.dtypes)
        return (dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims))

    @requires('posterior_model')
    @requires(['observed_data', 'constant_data'])
    def observed_and_constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        obs_const_dict = {}
        for group_name in ('observed_data', 'constant_data'):
            names = getattr(self, group_name)
            if names is None:
                continue
            names = [names] if isinstance(names, str) else names
            data = OrderedDict()
            for key in names:
                vals = np.atleast_1d(posterior_model.data[key])
                val_dims = dims.get(key)
                val_dims, coords = generate_dims_coords(vals.shape, key, dims=val_dims, coords=self.coords)
                data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
            obs_const_dict[group_name] = xr.Dataset(data_vars=data, attrs=make_attrs(library=self.stan))
        return obs_const_dict

    @requires('posterior_model')
    @requires('predictions_constant_data')
    def predictions_constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        names = self.predictions_constant_data
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior_model.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(vals.shape, key, dims=val_dims, coords=self.coords)
            data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=data, attrs=make_attrs(library=self.stan))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `fit`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        obs_const_dict = self.observed_and_constant_data_to_xarray()
        predictions_const_data = self.predictions_constant_data_to_xarray()
        return InferenceData(save_warmup=self.save_warmup, **{'posterior': self.posterior_to_xarray(), 'sample_stats': self.sample_stats_to_xarray(), 'log_likelihood': self.log_likelihood_to_xarray(), 'posterior_predictive': self.posterior_predictive_to_xarray(), 'predictions': self.predictions_to_xarray(), 'prior': self.prior_to_xarray(), 'sample_stats_prior': self.sample_stats_prior_to_xarray(), 'prior_predictive': self.prior_predictive_to_xarray(), **({} if obs_const_dict is None else obs_const_dict), **({} if predictions_const_data is None else {'predictions_constant_data': predictions_const_data})})