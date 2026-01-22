import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class CurvatureOutputSpec(TraitedSpec):
    out_mean = File(exists=False, desc='Mean curvature output file')
    out_gauss = File(exists=False, desc='Gaussian curvature output file')