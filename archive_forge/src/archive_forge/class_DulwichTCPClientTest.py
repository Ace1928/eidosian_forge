import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
class DulwichTCPClientTest(CompatTestCase, DulwichClientTestBase):

    def setUp(self):
        CompatTestCase.setUp(self)
        DulwichClientTestBase.setUp(self)
        if check_for_daemon(limit=1):
            raise SkipTest('git-daemon was already running on port %s' % protocol.TCP_GIT_PORT)
        fd, self.pidfile = tempfile.mkstemp(prefix='dulwich-test-git-client', suffix='.pid')
        os.fdopen(fd).close()
        args = [_DEFAULT_GIT, 'daemon', '--verbose', '--export-all', '--pid-file=%s' % self.pidfile, '--base-path=%s' % self.gitroot, '--enable=receive-pack', '--enable=upload-archive', '--listen=localhost', '--reuseaddr', self.gitroot]
        self.process = subprocess.Popen(args, cwd=self.gitroot, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not check_for_daemon():
            raise SkipTest('git-daemon failed to start')

    def tearDown(self):
        with open(self.pidfile) as f:
            pid = int(f.read().strip())
        if sys.platform == 'win32':
            PROCESS_TERMINATE = 1
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            ctypes.windll.kernel32.TerminateProcess(handle, -1)
            ctypes.windll.kernel32.CloseHandle(handle)
        else:
            with suppress(OSError):
                os.kill(pid, signal.SIGKILL)
                os.unlink(self.pidfile)
        self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()
        DulwichClientTestBase.tearDown(self)
        CompatTestCase.tearDown(self)

    def _client(self):
        return client.TCPGitClient('localhost')

    def _build_path(self, path):
        return path
    if sys.platform == 'win32':

        @expectedFailure
        def test_fetch_pack_no_side_band_64k(self):
            DulwichClientTestBase.test_fetch_pack_no_side_band_64k(self)

    def test_send_remove_branch(self):
        self.skipTest('skip flaky test; see #1015')