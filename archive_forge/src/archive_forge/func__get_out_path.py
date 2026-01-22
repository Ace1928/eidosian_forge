import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
def _get_out_path(self, meta, idx=None):
    """Return the output path for the generated Nifti."""
    if self.inputs.out_format:
        out_fmt = self.inputs.out_format
    else:
        out_fmt = []
        if idx is not None:
            out_fmt.append('%03d' % idx)
        if 'SeriesNumber' in meta:
            out_fmt.append('%(SeriesNumber)03d')
        if 'ProtocolName' in meta:
            out_fmt.append('%(ProtocolName)s')
        elif 'SeriesDescription' in meta:
            out_fmt.append('%(SeriesDescription)s')
        else:
            out_fmt.append('sequence')
        out_fmt = '-'.join(out_fmt)
    out_fn = out_fmt % meta + self.inputs.out_ext
    out_fn = sanitize_path_comp(out_fn)
    out_path = os.getcwd()
    if isdefined(self.inputs.out_path):
        out_path = op.abspath(self.inputs.out_path)
        try:
            os.makedirs(out_path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and op.isdir(out_path):
                pass
            else:
                raise
    return op.join(out_path, out_fn)