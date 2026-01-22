import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def _make_out_path(self, output_dir, strip_dir, src_name):
    base, ext = os.path.splitext(src_name)
    base = self._make_relative(base)
    try:
        new_ext = self.out_extensions[ext]
    except LookupError:
        raise UnknownFileError("unknown file type '{}' (from '{}')".format(ext, src_name))
    if strip_dir:
        base = os.path.basename(base)
    return os.path.join(output_dir, base + new_ext)