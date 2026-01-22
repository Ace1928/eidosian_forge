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
def _select_scheme(ob, name):
    scheme = _inject_headers(name, _load_scheme(_resolve_scheme(name)))
    vars(ob).update(_remove_set(ob, _scheme_attrs(scheme)))