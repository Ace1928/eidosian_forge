import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class MergeInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    axis = traits.Enum('vstack', 'hstack', usedefault=True, desc='direction in which to merge, hstack requires same number of elements in each input')
    no_flatten = traits.Bool(False, usedefault=True, desc='append to outlist instead of extending in vstack mode')
    ravel_inputs = traits.Bool(False, usedefault=True, desc='ravel inputs when no_flatten is False')