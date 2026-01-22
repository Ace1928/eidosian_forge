import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyCorrectInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, desc='4D input file', argstr='%s', position=0, mandatory=True)
    out_file = File(desc='4D output file', argstr='%s', position=1, name_source=['in_file'], name_template='%s_edc', output_name='eddy_corrected')
    ref_num = traits.Int(0, argstr='%d', position=2, desc='reference number', mandatory=True, usedefault=True)