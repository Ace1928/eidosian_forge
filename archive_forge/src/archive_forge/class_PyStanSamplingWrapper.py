from typing import Union
from ..data import from_cmdstanpy, from_pystan
from .base import SamplingWrapper
class PyStanSamplingWrapper(StanSamplingWrapper):
    """PyStan (3.0+) sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``PyStanSamplingWrapper`` usage can be found
    in the :ref:`pystan_refitting` notebook.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    """

    def sample(self, modified_observed_data):
        """Rebuild and resample the PyStan model on modified_observed_data."""
        import stan
        self.model: Union[str, stan.Model]
        if isinstance(self.model, str):
            program_code = self.model
        else:
            program_code = self.model.program_code
        self.model = stan.build(program_code, data=modified_observed_data)
        fit = self.model.sample(**self.sample_kwargs)
        return fit