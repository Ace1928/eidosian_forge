import typing as tp
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import xarray
from .inference_data import InferenceData
from ..rcparams import rcParams
from .base import dict_to_dataset
class PyJAGSConverter:
    """Encapsulate PyJAGS specific logic."""

    def __init__(self, *, posterior: tp.Optional[tp.Mapping[str, np.ndarray]]=None, prior: tp.Optional[tp.Mapping[str, np.ndarray]]=None, log_likelihood: tp.Optional[tp.Union[str, tp.List[str], tp.Tuple[str, ...], tp.Mapping[str, str]]]=None, coords=None, dims=None, save_warmup: tp.Optional[bool]=None, warmup_iterations: int=0) -> None:
        self.posterior: tp.Optional[tp.Mapping[str, np.ndarray]]
        self.log_likelihood: tp.Optional[tp.Dict[str, np.ndarray]]
        if log_likelihood is not None and posterior is not None:
            posterior_copy = dict(posterior)
            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]
            if isinstance(log_likelihood, (list, tuple)):
                log_likelihood = {name: name for name in log_likelihood}
            self.log_likelihood = {obs_var_name: posterior_copy.pop(log_like_name) for obs_var_name, log_like_name in log_likelihood.items()}
            self.posterior = posterior_copy
        else:
            self.posterior = posterior
            self.log_likelihood = None
        self.prior = prior
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams['data.save_warmup'] if save_warmup is None else save_warmup
        self.warmup_iterations = warmup_iterations
        import pyjags
        self.pyjags = pyjags

    def _pyjags_samples_to_xarray(self, pyjags_samples: tp.Mapping[str, np.ndarray]) -> tp.Tuple[xarray.Dataset, xarray.Dataset]:
        data, data_warmup = get_draws(pyjags_samples=pyjags_samples, warmup_iterations=self.warmup_iterations, warmup=self.save_warmup)
        return (dict_to_dataset(data, library=self.pyjags, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.pyjags, coords=self.coords, dims=self.dims))

    def posterior_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
        """Extract posterior samples from fit."""
        if self.posterior is None:
            return None
        return self._pyjags_samples_to_xarray(self.posterior)

    def prior_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
        """Extract posterior samples from fit."""
        if self.prior is None:
            return None
        return self._pyjags_samples_to_xarray(self.prior)

    def log_likelihood_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
        """Extract log likelihood samples from fit."""
        if self.log_likelihood is None:
            return None
        return self._pyjags_samples_to_xarray(self.log_likelihood)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        save_warmup = self.save_warmup and self.warmup_iterations > 0
        idata_dict = {'posterior': self.posterior_to_xarray(), 'prior': self.prior_to_xarray(), 'log_likelihood': self.log_likelihood_to_xarray(), 'save_warmup': save_warmup}
        return InferenceData(**idata_dict)