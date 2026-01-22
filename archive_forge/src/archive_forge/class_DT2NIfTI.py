import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class DT2NIfTI(CommandLine):
    """
    Converts camino tensor data to NIfTI format

    Reads Camino diffusion tensors, and converts them to NIFTI format as three .nii files.
    """
    _cmd = 'dt2nii'
    input_spec = DT2NIfTIInputSpec
    output_spec = DT2NIfTIOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        output_root = self._gen_outputroot()
        outputs['dt'] = os.path.abspath(output_root + 'dt.nii')
        outputs['exitcode'] = os.path.abspath(output_root + 'exitcode.nii')
        outputs['lns0'] = os.path.abspath(output_root + 'lns0.nii')
        return outputs

    def _gen_outfilename(self):
        return self._gen_outputroot()

    def _gen_outputroot(self):
        output_root = self.inputs.output_root
        if not isdefined(output_root):
            output_root = self._gen_filename('output_root')
        return output_root

    def _gen_filename(self, name):
        if name == 'output_root':
            _, filename, _ = split_filename(self.inputs.in_file)
            filename = filename + '_'
        return filename