import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class TalairachAVIOutputSpec(TraitedSpec):
    out_file = File(exists=False, desc='The output transform for TalairachAVI')
    out_log = File(exists=False, desc='The output log file for TalairachAVI')
    out_txt = File(exists=False, desc='The output text file for TaliarachAVI')