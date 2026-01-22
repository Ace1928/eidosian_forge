import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _get_outputfilenames(self, inverse=False):
    output_filename = None
    if not inverse:
        if isdefined(self.inputs.output_warped_image) and self.inputs.output_warped_image:
            output_filename = self.inputs.output_warped_image
            if isinstance(output_filename, bool):
                output_filename = '%s_Warped.nii.gz' % self.inputs.output_transform_prefix
        return output_filename
    inv_output_filename = None
    if isdefined(self.inputs.output_inverse_warped_image) and self.inputs.output_inverse_warped_image:
        inv_output_filename = self.inputs.output_inverse_warped_image
        if isinstance(inv_output_filename, bool):
            inv_output_filename = '%s_InverseWarped.nii.gz' % self.inputs.output_transform_prefix
    return inv_output_filename