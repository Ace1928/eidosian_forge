import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeFractionalAnisotropy(StdOutCommandLine):
    """
    Computes the fractional anisotropy of tensors.

    Reads diffusion tensor (single, two-tensor or three-tensor) data from the standard input,
    computes the fractional anisotropy (FA) of each tensor and outputs the results to the
    standard output. For multiple-tensor data the program outputs the FA of each tensor,
    so for three-tensor data, for example, the output contains three fractional anisotropy
    values per voxel.

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> fa = cmon.ComputeFractionalAnisotropy()
    >>> fa.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> fa.inputs.scheme_file = 'A.scheme'
    >>> fa.run()                  # doctest: +SKIP

    """
    _cmd = 'fa'
    input_spec = ComputeFractionalAnisotropyInputSpec
    output_spec = ComputeFractionalAnisotropyOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['fa'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_FA.Bdouble'