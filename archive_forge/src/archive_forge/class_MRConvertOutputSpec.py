import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRConvertOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image')
    json_export = File(exists=True, desc='exported data from an image header key-value pairs in a JSON file')
    out_bvec = File(exists=True, desc='export bvec file in FSL format')
    out_bval = File(exists=True, desc='export bvec file in FSL format')