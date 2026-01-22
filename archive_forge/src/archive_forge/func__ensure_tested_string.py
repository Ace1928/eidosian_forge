import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def _ensure_tested_string(self, option, tester, what, error_fmt, default=None):
    val = self._ensure_stringlike(option, what, default)
    if val is not None and (not tester(val)):
        raise DistutilsOptionError(("error in '%s' option: " + error_fmt) % (option, val))