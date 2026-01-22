import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class MergeOutputSpec(TraitedSpec):
    out = traits.List(desc='Merged output')