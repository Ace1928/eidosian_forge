import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Tkregister2(FSCommand):
    """

    Examples
    --------
    Get transform matrix between orig (*tkRAS*) and native (*scannerRAS*)
    coordinates in Freesurfer. Implements the first step of mapping surfaces
    to native space in `this guide
    <http://surfer.nmr.mgh.harvard.edu/fswiki/FsAnat-to-NativeAnat>`__.

    >>> from nipype.interfaces.freesurfer import Tkregister2
    >>> tk2 = Tkregister2(reg_file='T1_to_native.dat')
    >>> tk2.inputs.moving_image = 'T1.mgz'
    >>> tk2.inputs.target_image = 'structural.nii'
    >>> tk2.inputs.reg_header = True
    >>> tk2.cmdline
    'tkregister2 --mov T1.mgz --noedit --reg T1_to_native.dat --regheader --targ structural.nii'
    >>> tk2.run() # doctest: +SKIP

    The example below uses tkregister2 without the manual editing
    stage to convert FSL-style registration matrix (.mat) to
    FreeSurfer-style registration matrix (.dat)

    >>> from nipype.interfaces.freesurfer import Tkregister2
    >>> tk2 = Tkregister2()
    >>> tk2.inputs.moving_image = 'epi.nii'
    >>> tk2.inputs.fsl_in_matrix = 'flirt.mat'
    >>> tk2.cmdline
    'tkregister2 --fsl flirt.mat --mov epi.nii --noedit --reg register.dat'
    >>> tk2.run() # doctest: +SKIP
    """
    _cmd = 'tkregister2'
    input_spec = Tkregister2InputSpec
    output_spec = Tkregister2OutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'lta_in' and self.inputs.invert_lta_in:
            spec = '--lta-inv %s'
        if name in ('fsl_out', 'lta_out') and value is True:
            value = self._list_outputs()[name]
        return super(Tkregister2, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        reg_file = os.path.abspath(self.inputs.reg_file)
        outputs['reg_file'] = reg_file
        cwd = os.getcwd()
        fsl_out = self.inputs.fsl_out
        if isdefined(fsl_out):
            if fsl_out is True:
                outputs['fsl_file'] = fname_presuffix(reg_file, suffix='.mat', newpath=cwd, use_ext=False)
            else:
                outputs['fsl_file'] = os.path.abspath(self.inputs.fsl_out)
        lta_out = self.inputs.lta_out
        if isdefined(lta_out):
            if lta_out is True:
                outputs['lta_file'] = fname_presuffix(reg_file, suffix='.lta', newpath=cwd, use_ext=False)
            else:
                outputs['lta_file'] = os.path.abspath(self.inputs.lta_out)
        return outputs

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            return os.path.abspath(self.inputs.out_file)
        else:
            _, name, ext = split_filename(self.inputs.in_file)
            return os.path.abspath(name + '_smoothed' + ext)