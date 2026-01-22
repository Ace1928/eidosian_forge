import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class TalairachQCInputSpec(FSTraitedSpec):
    log_file = File(argstr='%s', mandatory=True, exists=True, position=0, desc='The log file for TalairachQC')