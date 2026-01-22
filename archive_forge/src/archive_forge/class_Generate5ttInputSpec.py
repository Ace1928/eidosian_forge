import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class Generate5ttInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum('fsl', 'gif', 'freesurfer', argstr='%s', position=-3, mandatory=True, desc='tissue segmentation algorithm')
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input image')
    out_file = File(argstr='%s', mandatory=True, position=-1, desc='output image')