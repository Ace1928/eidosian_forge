import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
class _FunctionalBase(object):

    def setUp(self):
        super(_FunctionalBase, self).setUp()
        tmpdir = self.useFixture(fixtures.TempDir()).path
        self.config_file = os.path.join(tmpdir, 'rootwrap.conf')
        self.later_cmd = os.path.join(tmpdir, 'later_install_cmd')
        filters_dir = os.path.join(tmpdir, 'filters.d')
        filters_file = os.path.join(tmpdir, 'filters.d', 'test.filters')
        os.mkdir(filters_dir)
        with open(self.config_file, 'w') as f:
            f.write('[DEFAULT]\nfilters_path=%s\ndaemon_timeout=10\nexec_dirs=/bin' % (filters_dir,))
        with open(filters_file, 'w') as f:
            f.write('[Filters]\necho: CommandFilter, /bin/echo, root\ncat: CommandFilter, /bin/cat, root\nsh: CommandFilter, /bin/sh, root\nid: CommandFilter, /usr/bin/id, nobody\nunknown_cmd: CommandFilter, /unknown/unknown_cmd, root\nlater_install_cmd: CommandFilter, %s, root\n' % self.later_cmd)

    def _test_run_once(self, expect_byte=True):
        code, out, err = self.execute(['echo', 'teststr'])
        self.assertEqual(0, code)
        if expect_byte:
            expect_out = b'teststr\n'
            expect_err = b''
        else:
            expect_out = 'teststr\n'
            expect_err = ''
        self.assertEqual(expect_out, out)
        self.assertEqual(expect_err, err)

    def _test_run_with_stdin(self, expect_byte=True):
        code, out, err = self.execute(['cat'], stdin=b'teststr')
        self.assertEqual(0, code)
        if expect_byte:
            expect_out = b'teststr'
            expect_err = b''
        else:
            expect_out = 'teststr'
            expect_err = ''
        self.assertEqual(expect_out, out)
        self.assertEqual(expect_err, err)

    def test_run_with_path(self):
        code, out, err = self.execute(['/bin/echo', 'teststr'])
        self.assertEqual(0, code)

    def test_run_with_bogus_path(self):
        code, out, err = self.execute(['/home/bob/bin/echo', 'teststr'])
        self.assertEqual(cmd.RC_UNAUTHORIZED, code)

    def test_run_command_not_found(self):
        code, out, err = self.execute(['unknown_cmd'])
        self.assertEqual(cmd.RC_NOEXECFOUND, code)

    def test_run_unauthorized_command(self):
        code, out, err = self.execute(['unauthorized_cmd'])
        self.assertEqual(cmd.RC_UNAUTHORIZED, code)

    def test_run_as(self):
        if os.getuid() != 0:
            self.skip('Test requires root (for setuid)')
        code, out, err = self.execute(['id', '-u'])
        self.assertEqual('%s\n' % pwd.getpwnam('nobody').pw_uid, out)
        code, out, err = self.execute(['sh', '-c', 'id -u'])
        self.assertEqual('0\n', out)