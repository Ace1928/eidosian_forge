import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class LTAConvertOutputSpec(TraitedSpec):
    out_lta = File(exists=True, desc='output linear transform (LTA Freesurfer format)')
    out_fsl = File(exists=True, desc='output transform in FSL format')
    out_mni = File(exists=True, desc='output transform in MNI/XFM format')
    out_reg = File(exists=True, desc='output transform in reg dat format')
    out_itk = File(exists=True, desc='output transform in ITK format')