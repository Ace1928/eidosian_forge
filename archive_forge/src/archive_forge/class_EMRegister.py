import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class EMRegister(FSCommandOpenMP):
    """This program creates a transform in lta format

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import EMRegister
    >>> register = EMRegister()
    >>> register.inputs.in_file = 'norm.mgz'
    >>> register.inputs.template = 'aseg.mgz'
    >>> register.inputs.out_file = 'norm_transform.lta'
    >>> register.inputs.skull = True
    >>> register.inputs.nbrspacing = 9
    >>> register.cmdline
    'mri_em_register -uns 9 -skull norm.mgz aseg.mgz norm_transform.lta'
    """
    _cmd = 'mri_em_register'
    input_spec = EMRegisterInputSpec
    output_spec = EMRegisterOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs