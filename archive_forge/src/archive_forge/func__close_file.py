import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
def _close_file(self):
    self._file.flush()
    self._file.close()
    self._file = None
    self._file_path = None
    self._file_dev = -1
    self._file_ino = -1