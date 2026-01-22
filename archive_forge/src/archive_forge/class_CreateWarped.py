import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class CreateWarped(SPMCommand):
    """Apply a flow field estimated by DARTEL to create warped images

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=190

    Examples
    --------
    >>> import nipype.interfaces.spm as spm
    >>> create_warped = spm.CreateWarped()
    >>> create_warped.inputs.image_files = ['rc1s1.nii', 'rc1s2.nii']
    >>> create_warped.inputs.flowfield_files = ['u_rc1s1_Template.nii', 'u_rc1s2_Template.nii']
    >>> create_warped.run() # doctest: +SKIP

    """
    input_spec = CreateWarpedInputSpec
    output_spec = CreateWarpedOutputSpec
    _jobtype = 'tools'
    _jobname = 'dartel'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['image_files']:
            return scans_for_fnames(val, keep4d=True, separate_sessions=True)
        if opt in ['flowfield_files']:
            return scans_for_fnames(val, keep4d=True)
        else:
            return super(CreateWarped, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['warped_files'] = []
        for filename in self.inputs.image_files:
            pth, base, ext = split_filename(filename)
            if isdefined(self.inputs.modulate) and self.inputs.modulate:
                outputs['warped_files'].append(os.path.realpath('mw%s%s' % (base, ext)))
            else:
                outputs['warped_files'].append(os.path.realpath('w%s%s' % (base, ext)))
        return outputs