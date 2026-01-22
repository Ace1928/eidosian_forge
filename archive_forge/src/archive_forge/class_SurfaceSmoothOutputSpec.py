import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceSmoothOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='smoothed surface file')