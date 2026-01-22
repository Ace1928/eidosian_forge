import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class DWIExtract(MRTrix3Base):
    """
    Extract diffusion-weighted volumes, b=0 volumes, or certain shells from a
    DWI dataset

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> dwiextract = mrt.DWIExtract()
    >>> dwiextract.inputs.in_file = 'dwi.mif'
    >>> dwiextract.inputs.bzero = True
    >>> dwiextract.inputs.out_file = 'b0vols.mif'
    >>> dwiextract.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> dwiextract.cmdline                             # doctest: +ELLIPSIS
    'dwiextract -bzero -fslgrad bvecs bvals dwi.mif b0vols.mif'
    >>> dwiextract.run()                               # doctest: +SKIP
    """
    _cmd = 'dwiextract'
    input_spec = DWIExtractInputSpec
    output_spec = DWIExtractOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs