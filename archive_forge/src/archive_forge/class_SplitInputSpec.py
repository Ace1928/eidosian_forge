import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class SplitInputSpec(BaseInterfaceInputSpec):
    inlist = traits.List(traits.Any, mandatory=True, desc='list of values to split')
    splits = traits.List(traits.Int, mandatory=True, desc='Number of outputs in each split - should add to number of inputs')
    squeeze = traits.Bool(False, usedefault=True, desc='unfold one-element splits removing the list')