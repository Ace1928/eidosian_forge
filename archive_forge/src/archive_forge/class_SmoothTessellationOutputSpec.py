import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SmoothTessellationOutputSpec(TraitedSpec):
    """
    This program smooths the tessellation of a surface using 'mris_smooth'
    """
    surface = File(exists=True, desc='Smoothed surface file.')