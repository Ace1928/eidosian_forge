import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class RobustRegister(FSCommand):
    """Perform intramodal linear registration (translation and rotation) using
    robust statistics.

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import RobustRegister
    >>> reg = RobustRegister()
    >>> reg.inputs.source_file = 'structural.nii'
    >>> reg.inputs.target_file = 'T1.nii'
    >>> reg.inputs.auto_sens = True
    >>> reg.inputs.init_orient = True
    >>> reg.cmdline # doctest: +ELLIPSIS
    'mri_robust_register --satit --initorient --lta .../structural_robustreg.lta --mov structural.nii --dst T1.nii'

    References
    ----------
    Reuter, M, Rosas, HD, and Fischl, B, (2010). Highly Accurate Inverse
        Consistent Registration: A Robust Approach.  Neuroimage 53(4) 1181-96.

    """
    _cmd = 'mri_robust_register'
    input_spec = RobustRegisterInputSpec
    output_spec = RobustRegisterOutputSpec

    def _format_arg(self, name, spec, value):
        options = ('out_reg_file', 'registered_file', 'weights_file', 'half_source', 'half_targ', 'half_weights', 'half_source_xfm', 'half_targ_xfm')
        if name in options and isinstance(value, bool):
            value = self._list_outputs()[name]
        return super(RobustRegister, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        cwd = os.getcwd()
        prefixes = dict(src=self.inputs.source_file, trg=self.inputs.target_file)
        suffixes = dict(out_reg_file=('src', '_robustreg.lta', False), registered_file=('src', '_robustreg', True), weights_file=('src', '_robustweights', True), half_source=('src', '_halfway', True), half_targ=('trg', '_halfway', True), half_weights=('src', '_halfweights', True), half_source_xfm=('src', '_robustxfm.lta', False), half_targ_xfm=('trg', '_robustxfm.lta', False))
        for name, sufftup in list(suffixes.items()):
            value = getattr(self.inputs, name)
            if value:
                if value is True:
                    outputs[name] = fname_presuffix(prefixes[sufftup[0]], suffix=sufftup[1], newpath=cwd, use_ext=sufftup[2])
                else:
                    outputs[name] = os.path.abspath(value)
        return outputs