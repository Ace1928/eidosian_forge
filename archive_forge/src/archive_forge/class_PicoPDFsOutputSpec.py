import os
from ...utils.filemanip import split_filename
from ..base import (
class PicoPDFsOutputSpec(TraitedSpec):
    pdfs = File(exists=True, desc='path/name of 4D volume in voxel order')