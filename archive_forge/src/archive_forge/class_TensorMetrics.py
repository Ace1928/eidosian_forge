import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TensorMetrics(CommandLine):
    """
    Compute metrics from tensors


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> comp = mrt.TensorMetrics()
    >>> comp.inputs.in_file = 'dti.mif'
    >>> comp.inputs.out_fa = 'fa.mif'
    >>> comp.cmdline                               # doctest: +ELLIPSIS
    'tensor2metric -num 1 -fa fa.mif dti.mif'
    >>> comp.run()                                 # doctest: +SKIP
    """
    _cmd = 'tensor2metric'
    input_spec = TensorMetricsInputSpec
    output_spec = TensorMetricsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for k in list(outputs.keys()):
            if isdefined(getattr(self.inputs, k)):
                outputs[k] = op.abspath(getattr(self.inputs, k))
        return outputs