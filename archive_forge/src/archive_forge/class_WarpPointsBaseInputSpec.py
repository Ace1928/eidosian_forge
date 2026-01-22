import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPointsBaseInputSpec(CommandLineInputSpec):
    in_coords = File(exists=True, position=-1, argstr='%s', mandatory=True, desc='filename of file containing coordinates')
    xfm_file = File(exists=True, argstr='-xfm %s', xor=['warp_file'], desc='filename of affine transform (e.g. source2dest.mat)')
    warp_file = File(exists=True, argstr='-warp %s', xor=['xfm_file'], desc='filename of warpfield (e.g. intermediate2dest_warp.nii.gz)')
    coord_vox = traits.Bool(True, argstr='-vox', xor=['coord_mm'], desc='all coordinates in voxels - default')
    coord_mm = traits.Bool(False, argstr='-mm', xor=['coord_vox'], desc='all coordinates in mm')
    out_file = File(name_source='in_coords', name_template='%s_warped', output_name='out_file', desc='output file name')