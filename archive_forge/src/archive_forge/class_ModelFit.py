import os
from ...utils.filemanip import split_filename
from ..base import (
class ModelFit(StdOutCommandLine):
    """
    Fits models of the spin-displacement density to diffusion MRI measurements.

    This is an interface to various model fitting routines for diffusion MRI data that
    fit models of the spin-displacement density function. In particular, it will fit the
    diffusion tensor to a set of measurements as well as various other models including
    two or three-tensor models. The program can read input data from a file or can
    generate synthetic data using various test functions for testing and simulations.

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> fit = cmon.ModelFit()
    >>> fit.model = 'dt'
    >>> fit.inputs.scheme_file = 'A.scheme'
    >>> fit.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> fit.run()                  # doctest: +SKIP

    """
    _cmd = 'modelfit'
    input_spec = ModelFitInputSpec
    output_spec = ModelFitOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['fitted_data'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_fit.Bdouble'