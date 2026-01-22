import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TransformFSLConvertInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=1, desc='FLIRT input image')
    reference = File(exists=True, argstr='%s', mandatory=True, position=2, desc='FLIRT reference image')
    in_transform = File(exists=True, argstr='%s', mandatory=True, position=0, desc='FLIRT output transformation matrix')
    out_transform = File('transform_mrtrix.txt', argstr='%s', mandatory=True, position=-1, usedefault=True, desc="output transformed affine in mrtrix3's format")
    flirt_import = traits.Bool(True, argstr='flirt_import', mandatory=True, usedefault=True, position=-2, desc="import transform from FSL's FLIRT.")