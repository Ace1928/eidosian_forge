import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class FilterTracksOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output filtered tracks')