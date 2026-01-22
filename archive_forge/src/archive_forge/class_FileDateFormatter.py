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
class FileDateFormatter:

    def __init__(self, datetime=None):
        self.datetime = datetime or aware_now()

    def __format__(self, spec):
        if not spec:
            spec = '%Y-%m-%d_%H-%M-%S_%f'
        return self.datetime.__format__(spec)