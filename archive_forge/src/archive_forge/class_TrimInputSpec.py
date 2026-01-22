import os
import nibabel as nb
import numpy as np
from ...utils.filemanip import split_filename, fname_presuffix
from .base import NipyBaseInterface, have_nipy
from ..base import (
class TrimInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='EPI image to trim')
    begin_index = traits.Int(0, usedefault=True, desc='first volume')
    end_index = traits.Int(0, usedefault=True, desc='last volume indexed as in python (and 0 for last)')
    out_file = File(desc='output filename')
    suffix = traits.Str('_trim', usedefault=True, desc='suffix for out_file to use if no out_file provided')