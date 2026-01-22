import functools
import glob
import os
import re
import subprocess
import sys
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_aix as cext
from . import _psutil_posix as cext_posix
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import conn_to_ntuple
from ._common import get_procfs_path
from ._common import memoize_when_activated
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
Wrapper class around underlying C implementation.