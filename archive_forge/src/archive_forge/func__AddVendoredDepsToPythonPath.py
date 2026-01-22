from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import pkgutil
import sys
import tempfile
import gslib.exception  # pylint: disable=g-import-not-at-top
from gslib.utils.version_check import check_python_version_support
def _AddVendoredDepsToPythonPath():
    """Fix our Python path so that it correctly finds our vendored libraries."""
    vendored_path = os.path.join(GSLIB_DIR, 'vendored')
    vendored_lib_dirs = [('boto', ''), ('oauth2client', '')]
    for libdir, subdir in vendored_lib_dirs:
        sys.path.insert(0, os.path.join(vendored_path, libdir, subdir))
    sys.path.append(os.path.join(vendored_path, 'boto', 'tests', 'integration', 's3'))