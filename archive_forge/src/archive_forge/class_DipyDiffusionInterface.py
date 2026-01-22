import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
class DipyDiffusionInterface(DipyBaseInterface):
    """A base interface for py:mod:`dipy` computations."""
    input_spec = DipyBaseInterfaceInputSpec

    def _get_gradient_table(self):
        bval = np.loadtxt(self.inputs.in_bval)
        bvec = np.loadtxt(self.inputs.in_bvec).T
        from dipy.core.gradients import gradient_table
        gtab = gradient_table(bval, bvec)
        gtab.b0_threshold = self.inputs.b0_thres
        return gtab

    def _gen_filename(self, name, ext=None):
        fname, fext = op.splitext(op.basename(self.inputs.in_file))
        if fext == '.gz':
            fname, fext2 = op.splitext(fname)
            fext = fext2 + fext
        if not isdefined(self.inputs.out_prefix):
            out_prefix = op.abspath(fname)
        else:
            out_prefix = self.inputs.out_prefix
        if ext is None:
            ext = fext
        return out_prefix + '_' + name + ext