import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class HARDIMatOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output matrix file')