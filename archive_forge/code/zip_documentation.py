import os
import stat
import sys
import tempfile
import time
import zipfile
from contextlib import closing
from .. import osutils
from ..export import _export_iter_entries
from ..trace import mutter
 Export this tree to a new zip file.

    `dest` will be created holding the contents of this tree; if it
    already exists, it will be overwritten".
    