import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class ShredderOutputSpec(TraitedSpec):
    shredded = File(exists=True, desc='Shredded binary data file')