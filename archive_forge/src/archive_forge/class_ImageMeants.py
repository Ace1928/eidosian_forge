import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ImageMeants(FSLCommand):
    """Use fslmeants for printing the average timeseries (intensities) to
    the screen (or saves to a file). The average is taken over all voxels
    in the mask (or all voxels in the image if no mask is specified)

    """
    _cmd = 'fslmeants'
    input_spec = ImageMeantsInputSpec
    output_spec = ImageMeantsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file, suffix='_ts', ext='.txt', change_ext=True)
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None