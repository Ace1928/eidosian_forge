import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class ODFReconOutputSpec(TraitedSpec):
    B0 = File(exists=True)
    DWI = File(exists=True)
    max = File(exists=True)
    ODF = File(exists=True)
    entropy = File()