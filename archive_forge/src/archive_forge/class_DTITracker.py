import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
class DTITracker(CommandLine):
    input_spec = DTITrackerInputSpec
    output_spec = DTITrackerOutputSpec
    _cmd = 'dti_tracker'

    def _run_interface(self, runtime):
        _, _, ext = split_filename(self.inputs.tensor_file)
        copyfile(self.inputs.tensor_file, os.path.abspath(self.inputs.input_data_prefix + '_tensor' + ext), copy=False)
        return super(DTITracker, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['track_file'] = os.path.abspath(self.inputs.output_file)
        if isdefined(self.inputs.output_mask) and self.inputs.output_mask:
            outputs['mask_file'] = os.path.abspath(self.inputs.output_mask)
        return outputs