import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class SH2Amp(CommandLine):
    """
    Sample spherical harmonics on a set of gradient orientations.  Useful for
    checking residuals of ODF estimates.


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> sh = mrt.SH2Amp()
    >>> sh.inputs.in_file = 'sh.mif'
    >>> sh.inputs.directions = 'grads.txt'
    >>> sh.cmdline
    'sh2amp sh.mif grads.txt sh_amp.mif'
    >>> sh.run()                                 # doctest: +SKIP
    """
    _cmd = 'sh2amp'
    input_spec = SH2AmpInputSpec
    output_spec = SH2AmpOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs