import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class RenameOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='softlink to original file with new name')