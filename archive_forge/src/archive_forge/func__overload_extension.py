import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
def _overload_extension(self, value, name=None):
    path, base, ext = split_filename(value)
    if ext.lower() not in ['.1d', '.1D', '.nii.gz', '.nii']:
        ext = ext + '.1D'
    return os.path.join(path, base + ext)