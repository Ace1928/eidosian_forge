import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class GenerateWhiteMatterMaskOutputSpec(TraitedSpec):
    WMprobabilitymap = File(exists=True, desc='WMprobabilitymap')