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
def _need_link(self, objects, output_file):
    """Return true if we need to relink the files listed in 'objects'
        to recreate 'output_file'.
        """
    if self.force:
        return True
    else:
        if self.dry_run:
            newer = newer_group(objects, output_file, missing='newer')
        else:
            newer = newer_group(objects, output_file)
        return newer