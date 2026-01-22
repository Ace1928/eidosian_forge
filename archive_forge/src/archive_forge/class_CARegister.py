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
class CARegister(FSCommandOpenMP):
    """Generates a multi-dimensional talairach transform from a gca file and talairach.lta file

    See Also
    --------
    For complete details, see the `FS Documentation
    <http://surfer.nmr.mgh.harvard.edu/fswiki/mri_ca_register>`__

    Examples
    --------
    >>> from nipype.interfaces import freesurfer
    >>> ca_register = freesurfer.CARegister()
    >>> ca_register.inputs.in_file = "norm.mgz"
    >>> ca_register.inputs.out_file = "talairach.m3z"
    >>> ca_register.cmdline
    'mri_ca_register norm.mgz talairach.m3z'

    """
    _cmd = 'mri_ca_register'
    input_spec = CARegisterInputSpec
    output_spec = CARegisterOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'l_files' and len(value) == 1:
            value.append('identity.nofile')
        return super(CARegister, self)._format_arg(name, spec, value)

    def _gen_fname(self, name):
        if name == 'out_file':
            return os.path.abspath('talairach.m3z')
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs