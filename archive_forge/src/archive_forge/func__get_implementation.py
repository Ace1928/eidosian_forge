import sys
import os
import contextlib
import sysconfig
import itertools
from distutils._log import log
from ..core import Command
from ..debug import DEBUG
from ..sysconfig import get_config_vars
from ..file_util import write_file
from ..util import convert_path, subst_vars, change_root
from ..util import get_platform
from ..errors import DistutilsOptionError, DistutilsPlatformError
from . import _framework_compat as fw
from .. import _collections
from site import USER_BASE
from site import USER_SITE
def _get_implementation():
    if hasattr(sys, 'pypy_version_info'):
        return 'PyPy'
    else:
        return 'Python'