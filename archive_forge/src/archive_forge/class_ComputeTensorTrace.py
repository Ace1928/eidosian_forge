import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeTensorTrace(StdOutCommandLine):
    """
    Computes the trace of tensors.

    Reads diffusion tensor (single, two-tensor or three-tensor) data from the standard input,
    computes the trace of each tensor, i.e., three times the mean diffusivity, and outputs
    the results to the standard output. For multiple-tensor data the program outputs the
    trace of each tensor, so for three-tensor data, for example, the output contains three
    values per voxel.

    Divide the output by three to get the mean diffusivity.

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> trace = cmon.ComputeTensorTrace()
    >>> trace.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> trace.inputs.scheme_file = 'A.scheme'
    >>> trace.run()                 # doctest: +SKIP

    """
    _cmd = 'trd'
    input_spec = ComputeTensorTraceInputSpec
    output_spec = ComputeTensorTraceOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['trace'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_TrD.img'