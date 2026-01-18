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
@staticmethod
def write_compress(path_in, path_out, opener, **kwargs):
    with opener(path_out, **kwargs) as f_comp:
        f_comp.write(path_in, os.path.basename(path_in))