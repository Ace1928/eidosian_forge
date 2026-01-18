import os
import shutil
import traceback
import unittest
import warnings
from contextlib import closing
import psutil
from psutil import BSD
from psutil import POSIX
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import super
from psutil._compat import u
from psutil.tests import APPVEYOR
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import INVALID_UNICODE_SUFFIX
from psutil.tests import PYPY
from psutil.tests import TESTFN_PREFIX
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import bind_unix_socket
from psutil.tests import chdir
from psutil.tests import copyload_shared_lib
from psutil.tests import create_py_exe
from psutil.tests import get_testfn
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import serialrun
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import terminate
def try_unicode(suffix):
    """Return True if both the fs and the subprocess module can
    deal with a unicode file name.
    """
    sproc = None
    testfn = get_testfn(suffix=suffix)
    try:
        safe_rmpath(testfn)
        create_py_exe(testfn)
        sproc = spawn_testproc(cmd=[testfn])
        shutil.copyfile(testfn, testfn + '-2')
        safe_rmpath(testfn + '-2')
    except (UnicodeEncodeError, IOError):
        return False
    else:
        return True
    finally:
        if sproc is not None:
            terminate(sproc)
        safe_rmpath(testfn)