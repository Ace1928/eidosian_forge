import errno
import functools
import os
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_osx as cext
from . import _psutil_posix as cext_posix
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import parse_environ_block
from ._common import usage_percent
from ._compat import PermissionError
from ._compat import ProcessLookupError
@wrap_exceptions
@memoize_when_activated
def _get_kinfo_proc(self):
    ret = cext.proc_kinfo_oneshot(self.pid)
    assert len(ret) == len(kinfo_proc_map)
    return ret