import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
def _parse_output(self, stdout, stderr):
    """Parse stdout / stderr and extract defects"""
    m = re.search('(?<=total defect index = )\\d+', stdout or stderr)
    if m is None:
        raise RuntimeError('Could not fetch defect index')
    self._defects = int(m.group())