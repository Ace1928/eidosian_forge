import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class AFNItoNIFTIInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dAFNItoNIFTI', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s.nii', desc='output image file name', argstr='-prefix %s', name_source='in_file', hash_files=False)
    float_ = traits.Bool(desc='Force the output dataset to be 32-bit floats. This option should be used when the input AFNI dataset has different float scale factors for different sub-bricks, an option that NIfTI-1.1 does not support.', argstr='-float')
    pure = traits.Bool(desc="Do NOT write an AFNI extension field into the output file. Only use this option if needed. You can also use the 'nifti_tool' program to strip extensions from a file.", argstr='-pure')
    denote = traits.Bool(desc='When writing the AFNI extension field, remove text notes that might contain subject identifying information.', argstr='-denote')
    oldid = traits.Bool(desc='Give the new dataset the input datasets AFNI ID code.', argstr='-oldid', xor=['newid'])
    newid = traits.Bool(desc='Give the new dataset a new AFNI ID code, to distinguish it from the input dataset.', argstr='-newid', xor=['oldid'])