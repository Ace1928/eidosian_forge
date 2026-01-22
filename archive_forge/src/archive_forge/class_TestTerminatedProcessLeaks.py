from __future__ import print_function
import functools
import os
import platform
import unittest
import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import ProcessLookupError
from psutil._compat import super
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate
class TestTerminatedProcessLeaks(TestProcessObjectLeaks):
    """Repeat the tests above looking for leaks occurring when dealing
    with terminated processes raising NoSuchProcess exception.
    The C functions are still invoked but will follow different code
    paths. We'll check those code paths.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.subp = spawn_testproc()
        cls.proc = psutil.Process(cls.subp.pid)
        cls.proc.kill()
        cls.proc.wait()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        terminate(cls.subp)

    def call(self, fun):
        try:
            fun()
        except psutil.NoSuchProcess:
            pass
    if WINDOWS:

        def test_kill(self):
            self.execute(self.proc.kill)

        def test_terminate(self):
            self.execute(self.proc.terminate)

        def test_suspend(self):
            self.execute(self.proc.suspend)

        def test_resume(self):
            self.execute(self.proc.resume)

        def test_wait(self):
            self.execute(self.proc.wait)

        def test_proc_info(self):

            def call():
                try:
                    return cext.proc_info(self.proc.pid)
                except ProcessLookupError:
                    pass
            self.execute(call)