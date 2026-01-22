import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class MRDeGibbsInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    axes = traits.ListInt(default_value=[0, 1], usedefault=True, sep=',', minlen=2, maxlen=2, argstr='-axes %s', desc='indicate the plane in which the data was acquired (axial = 0,1; coronal = 0,2; sagittal = 1,2')
    nshifts = traits.Int(default_value=20, usedefault=True, argstr='-nshifts %d', desc='discretization of subpixel spacing (default = 20)')
    minW = traits.Int(default_value=1, usedefault=True, argstr='-minW %d', desc='left border of window used for total variation (TV) computation (default = 1)')
    maxW = traits.Int(default_value=3, usedefault=True, argstr='-maxW %d', desc='right border of window used for total variation (TV) computation (default = 3)')
    out_file = File(name_template='%s_unr', name_source='in_file', keep_extension=True, argstr='%s', position=-1, desc='the output unringed DWI image')