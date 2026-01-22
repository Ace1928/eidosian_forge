import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class Image2Voxel(StdOutCommandLine):
    """
    Converts Analyze / NIFTI / MHA files to voxel order.

    Converts scanner-order data in a supported image format to voxel-order data.
    Either takes a 4D file (all measurements in single image)
    or a list of 3D images.

    Examples
    --------

    >>> import nipype.interfaces.camino as cmon
    >>> img2vox = cmon.Image2Voxel()
    >>> img2vox.inputs.in_file = '4d_dwi.nii'
    >>> img2vox.run()                  # doctest: +SKIP
    """
    _cmd = 'image2voxel'
    input_spec = Image2VoxelInputSpec
    output_spec = Image2VoxelOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['voxel_order'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '.B' + self.inputs.out_type