import contextlib
import errno
import functools
import os
from collections import defaultdict
from collections import namedtuple
from xml.etree import ElementTree
from . import _common
from . import _psposix
from . import _psutil_bsd as cext
from . import _psutil_posix as cext_posix
from ._common import FREEBSD
from ._common import NETBSD
from ._common import OPENBSD
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import debug
from ._common import memoize
from ._common import memoize_when_activated
from ._common import usage_percent
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import which
Return the number of file descriptors opened by this process.