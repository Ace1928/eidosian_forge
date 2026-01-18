import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import unittest
import psutil
import psutil.tests
from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import sh
def test_supports_ipv6(self):
    self.addCleanup(supports_ipv6.cache_clear)
    if supports_ipv6():
        with mock.patch('psutil._common.socket') as s:
            s.has_ipv6 = False
            supports_ipv6.cache_clear()
            assert not supports_ipv6()
        supports_ipv6.cache_clear()
        with mock.patch('psutil._common.socket.socket', side_effect=socket.error) as s:
            assert not supports_ipv6()
            assert s.called
        supports_ipv6.cache_clear()
        with mock.patch('psutil._common.socket.socket', side_effect=socket.gaierror) as s:
            assert not supports_ipv6()
            supports_ipv6.cache_clear()
            assert s.called
        supports_ipv6.cache_clear()
        with mock.patch('psutil._common.socket.socket.bind', side_effect=socket.gaierror) as s:
            assert not supports_ipv6()
            supports_ipv6.cache_clear()
            assert s.called
    else:
        with self.assertRaises(socket.error):
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            try:
                sock.bind(('::1', 0))
            finally:
                sock.close()