import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class FSL2SchemeOutputSpec(TraitedSpec):
    scheme = File(exists=True, desc='Scheme file')