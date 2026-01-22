import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpUtilsOutputSpec(TraitedSpec):
    out_file = File(desc='Name of output file, containing the warp as field or coefficients.')
    out_jacobian = File(desc='Name of output file, containing the map of the determinant of the Jacobian')