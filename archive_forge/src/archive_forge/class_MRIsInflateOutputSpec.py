import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsInflateOutputSpec(TraitedSpec):
    out_file = File(exists=False, desc='Output file for MRIsInflate')
    out_sulc = File(exists=False, desc='Output sulc file')