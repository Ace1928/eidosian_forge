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
def _fix_object_args(self, objects, output_dir):
    """Typecheck and fix up some arguments supplied to various methods.
        Specifically: ensure that 'objects' is a list; if output_dir is
        None, replace with self.output_dir.  Return fixed versions of
        'objects' and 'output_dir'.
        """
    if not isinstance(objects, (list, tuple)):
        raise TypeError("'objects' must be a list or tuple of strings")
    objects = list(objects)
    if output_dir is None:
        output_dir = self.output_dir
    elif not isinstance(output_dir, str):
        raise TypeError("'output_dir' must be a string or None")
    return (objects, output_dir)