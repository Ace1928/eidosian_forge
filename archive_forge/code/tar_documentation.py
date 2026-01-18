import os
import sys
import tarfile
from contextlib import closing
from io import BytesIO
from .. import errors, osutils
from ..export import _export_iter_entries
Export this tree to a new .tar.lzma file.

    `dest` will be created holding the contents of this tree; if it
    already exists, it will be clobbered, like with "tar -c".
    