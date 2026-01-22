import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ImageInfoOutputSpec(TraitedSpec):
    info = traits.Any(desc='output of mri_info')
    out_file = File(exists=True, desc='text file with image information')
    data_type = traits.String(desc='image data type')
    file_format = traits.String(desc='file format')
    TE = traits.String(desc='echo time (msec)')
    TR = traits.String(desc='repetition time(msec)')
    TI = traits.String(desc='inversion time (msec)')
    dimensions = traits.Tuple(desc='image dimensions (voxels)')
    vox_sizes = traits.Tuple(desc='voxel sizes (mm)')
    orientation = traits.String(desc='image orientation')
    ph_enc_dir = traits.String(desc='phase encode direction')