from typing import Union
from ..data import from_cmdstanpy, from_pystan
from .base import SamplingWrapper
class CmdStanPySamplingWrapper(StanSamplingWrapper):
    """CmdStanPy sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``CmdStanPySamplingWrapper`` usage can be found
    in the :ref:`cmdstanpy_refitting` notebook.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    """

    def __init__(self, data_file, **kwargs):
        """Initialize the CmdStanPySamplingWrapper.

        Parameters
        ----------
        data_file : str
            Filename on which to store the data for every refit.
            It's contents will be overwritten.
        """
        super().__init__(**kwargs)
        self.data_file = data_file

    def sample(self, modified_observed_data):
        """Resample cmdstanpy model on modified_observed_data."""
        from cmdstanpy import write_stan_json
        write_stan_json(self.data_file, modified_observed_data)
        fit = self.model.sample(**{**self.sample_kwargs, 'data': self.data_file})
        return fit