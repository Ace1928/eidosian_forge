import os.path as op
import nibabel as nb
import numpy as np
from looseversion import LooseVersion
from ... import logging
from ..base import traits, TraitedSpec, File, isdefined
from .base import (
def _gen_outfilename(self):
    fname, fext = op.splitext(op.basename(self.inputs.in_file))
    if fext == '.gz':
        fname, fext2 = op.splitext(fname)
        fext = fext2 + fext
    return op.abspath('%s_denoise%s' % (fname, fext))