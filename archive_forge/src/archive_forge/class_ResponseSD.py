import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class ResponseSD(MRTrix3Base):
    """
    Estimate response function(s) for spherical deconvolution using the specified algorithm.

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> resp = mrt.ResponseSD()
    >>> resp.inputs.in_file = 'dwi.mif'
    >>> resp.inputs.algorithm = 'tournier'
    >>> resp.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> resp.cmdline                               # doctest: +ELLIPSIS
    'dwi2response tournier -fslgrad bvecs bvals dwi.mif wm.txt'
    >>> resp.run()                                 # doctest: +SKIP

    # We can also pass in multiple harmonic degrees in the case of multi-shell
    >>> resp.inputs.max_sh = [6,8,10]
    >>> resp.cmdline
    'dwi2response tournier -fslgrad bvecs bvals -lmax 6,8,10 dwi.mif wm.txt'
    """
    _cmd = 'dwi2response'
    input_spec = ResponseSDInputSpec
    output_spec = ResponseSDOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wm_file'] = op.abspath(self.inputs.wm_file)
        if self.inputs.gm_file != Undefined:
            outputs['gm_file'] = op.abspath(self.inputs.gm_file)
        if self.inputs.csf_file != Undefined:
            outputs['csf_file'] = op.abspath(self.inputs.csf_file)
        return outputs