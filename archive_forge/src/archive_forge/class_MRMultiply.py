import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class MRMultiply(CommandLine):
    """
    Multiplies two images.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> MRmult = mrt.MRMultiply()
    >>> MRmult.inputs.in_files = ['dwi.mif', 'dwi_WMProb.mif']
    >>> MRmult.run()                                             # doctest: +SKIP
    """
    _cmd = 'mrmult'
    input_spec = MRMultiplyInputSpec
    output_spec = MRMultiplyOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_filename
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = op.abspath(self._gen_outfilename())
        else:
            outputs['out_file'] = op.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_files[0])
        return name + '_MRMult.mif'