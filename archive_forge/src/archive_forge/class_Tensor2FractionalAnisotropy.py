import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class Tensor2FractionalAnisotropy(CommandLine):
    """
    Generates a map of the fractional anisotropy in each voxel.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> tensor2FA = mrt.Tensor2FractionalAnisotropy()
    >>> tensor2FA.inputs.in_file = 'dwi_tensor.mif'
    >>> tensor2FA.run()                                 # doctest: +SKIP
    """
    _cmd = 'tensor2FA'
    input_spec = Tensor2FractionalAnisotropyInputSpec
    output_spec = Tensor2FractionalAnisotropyOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['FA'] = self.inputs.out_filename
        if not isdefined(outputs['FA']):
            outputs['FA'] = op.abspath(self._gen_outfilename())
        else:
            outputs['FA'] = op.abspath(outputs['FA'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_FA.mif'