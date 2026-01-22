import os
import nibabel as nb
import numpy as np
from ...utils.filemanip import split_filename, fname_presuffix
from .base import NipyBaseInterface, have_nipy
from ..base import (
class SpaceTimeRealignerInputSpec(BaseInterfaceInputSpec):
    in_file = InputMultiPath(File(exists=True), mandatory=True, min_ver='0.4.0.dev', desc='File to realign')
    tr = traits.Float(desc='TR in seconds', requires=['slice_times'])
    slice_times = traits.Either(traits.List(traits.Float()), traits.Enum('asc_alt_2', 'asc_alt_2_1', 'asc_alt_half', 'asc_alt_siemens', 'ascending', 'desc_alt_2', 'desc_alt_half', 'descending'), desc='Actual slice acquisition times.')
    slice_info = traits.Either(traits.Int, traits.List(min_len=2, max_len=2), desc='Single integer or length 2 sequence If int, the axis in `images` that is the slice axis.  In a 4D image, this will often be axis = 2.  If a 2 sequence, then elements are ``(slice_axis, slice_direction)``, where ``slice_axis`` is the slice axis in the image as above, and ``slice_direction`` is 1 if the slices were acquired slice 0 first, slice -1 last, or -1 if acquired slice -1 first, slice 0 last.  If `slice_info` is an int, assume ``slice_direction`` == 1.', requires=['slice_times'])