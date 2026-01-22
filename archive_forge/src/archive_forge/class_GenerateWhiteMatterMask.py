import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class GenerateWhiteMatterMask(CommandLine):
    """
    Generates a white matter probability mask from the DW images.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> genWM = mrt.GenerateWhiteMatterMask()
    >>> genWM.inputs.in_file = 'dwi.mif'
    >>> genWM.inputs.encoding_file = 'encoding.txt'
    >>> genWM.run()                                     # doctest: +SKIP
    """
    _cmd = 'gen_WM_mask'
    input_spec = GenerateWhiteMatterMaskInputSpec
    output_spec = GenerateWhiteMatterMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['WMprobabilitymap'] = op.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_WMProb_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_WMProb.mif'