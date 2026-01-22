from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def _get_library_dir():
    library_dir = None
    try:
        import sysconfig
        library_dir = sysconfig.get_path('purelib')
    except ImportError:
        pass
    if library_dir is None or not os_path_exists(library_dir):
        for path in sys.path:
            if os_path_exists(path) and os.path.basename(path) == 'site-packages':
                library_dir = path
                break
    if library_dir is None or not os_path_exists(library_dir):
        library_dir = os.path.dirname(os.__file__)
    return library_dir