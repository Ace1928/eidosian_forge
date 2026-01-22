import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='masked image')