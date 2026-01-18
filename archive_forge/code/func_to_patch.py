import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def to_patch(self):
    yield (b'c %(parent)d %(parent_pos)d %(child_pos)d %(num_lines)d\n' % self._as_dict())