import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
class PrlimitTestCase(test_base.BaseTestCase):
    SIMPLE_PROGRAM = [sys.executable, '-c', 'pass']

    def soft_limit(self, res, substract, default_limit):
        soft_limit, hard_limit = resource.getrlimit(res)
        if soft_limit <= 0:
            soft_limit = default_limit
        else:
            soft_limit -= substract
        return soft_limit

    def memory_limit(self, res):
        return self.soft_limit(res, 1024, 1024 ** 3)

    def limit_address_space(self):
        max_memory = self.memory_limit(resource.RLIMIT_AS)
        return processutils.ProcessLimits(address_space=max_memory)

    def test_simple(self):
        prlimit = self.limit_address_space()
        stdout, stderr = processutils.execute(*self.SIMPLE_PROGRAM, prlimit=prlimit)
        self.assertEqual('', stdout.rstrip())
        self.assertEqual(stderr.rstrip(), '')

    def check_limit(self, prlimit, resource, value):
        code = ';'.join(('import resource', 'print(resource.getrlimit(resource.%s))' % resource))
        args = [sys.executable, '-c', code]
        stdout, stderr = processutils.execute(*args, prlimit=prlimit)
        expected = (value, value)
        self.assertEqual(str(expected), stdout.rstrip())

    def test_address_space(self):
        prlimit = self.limit_address_space()
        self.check_limit(prlimit, 'RLIMIT_AS', prlimit.address_space)

    def test_core_size(self):
        size = self.soft_limit(resource.RLIMIT_CORE, 1, 1024)
        prlimit = processutils.ProcessLimits(core_file_size=size)
        self.check_limit(prlimit, 'RLIMIT_CORE', prlimit.core_file_size)

    def test_cpu_time(self):
        time = self.soft_limit(resource.RLIMIT_CPU, 1, 1024)
        prlimit = processutils.ProcessLimits(cpu_time=time)
        self.check_limit(prlimit, 'RLIMIT_CPU', prlimit.cpu_time)

    def test_data_size(self):
        max_memory = self.memory_limit(resource.RLIMIT_DATA)
        prlimit = processutils.ProcessLimits(data_size=max_memory)
        self.check_limit(prlimit, 'RLIMIT_DATA', max_memory)

    def test_file_size(self):
        size = self.soft_limit(resource.RLIMIT_FSIZE, 1, 1024)
        prlimit = processutils.ProcessLimits(file_size=size)
        self.check_limit(prlimit, 'RLIMIT_FSIZE', prlimit.file_size)

    def test_memory_locked(self):
        max_memory = self.memory_limit(resource.RLIMIT_MEMLOCK)
        prlimit = processutils.ProcessLimits(memory_locked=max_memory)
        self.check_limit(prlimit, 'RLIMIT_MEMLOCK', max_memory)

    def test_resident_set_size(self):
        max_memory = self.memory_limit(resource.RLIMIT_RSS)
        prlimit = processutils.ProcessLimits(resident_set_size=max_memory)
        self.check_limit(prlimit, 'RLIMIT_RSS', max_memory)

    def test_number_files(self):
        nfiles = self.soft_limit(resource.RLIMIT_NOFILE, 1, 1024)
        prlimit = processutils.ProcessLimits(number_files=nfiles)
        self.check_limit(prlimit, 'RLIMIT_NOFILE', nfiles)

    def test_number_processes(self):
        nprocs = self.soft_limit(resource.RLIMIT_NPROC, 1, 65535)
        prlimit = processutils.ProcessLimits(number_processes=nprocs)
        self.check_limit(prlimit, 'RLIMIT_NPROC', nprocs)

    def test_stack_size(self):
        max_memory = self.memory_limit(resource.RLIMIT_STACK)
        prlimit = processutils.ProcessLimits(stack_size=max_memory)
        self.check_limit(prlimit, 'RLIMIT_STACK', max_memory)

    def test_unsupported_prlimit(self):
        self.assertRaises(ValueError, processutils.ProcessLimits, xxx=33)

    def test_relative_path(self):
        prlimit = self.limit_address_space()
        program = sys.executable
        env = dict(os.environ)
        env['PATH'] = os.path.dirname(program)
        args = [os.path.basename(program), '-c', 'pass']
        processutils.execute(*args, prlimit=prlimit, env_variables=env)

    def test_execv_error(self):
        prlimit = self.limit_address_space()
        args = ['/missing_path/dont_exist/program']
        try:
            processutils.execute(*args, prlimit=prlimit)
        except processutils.ProcessExecutionError as exc:
            self.assertEqual(1, exc.exit_code)
            self.assertEqual('', exc.stdout)
            expected = '%s -m oslo_concurrency.prlimit: failed to execute /missing_path/dont_exist/program: ' % os.path.basename(sys.executable)
            self.assertIn(expected, exc.stderr)
        else:
            self.fail('ProcessExecutionError not raised')

    def test_setrlimit_error(self):
        prlimit = self.limit_address_space()
        higher_limit = prlimit.address_space + 1024
        args = [sys.executable, '-m', 'oslo_concurrency.prlimit', '--as=%s' % higher_limit, '--']
        args.extend(self.SIMPLE_PROGRAM)
        try:
            processutils.execute(*args, prlimit=prlimit)
        except processutils.ProcessExecutionError as exc:
            self.assertEqual(1, exc.exit_code)
            self.assertEqual('', exc.stdout)
            expected = '%s -m oslo_concurrency.prlimit: failed to set the AS resource limit: ' % os.path.basename(sys.executable)
            self.assertIn(expected, exc.stderr)
        else:
            self.fail('ProcessExecutionError not raised')

    @mock.patch.object(os, 'name', 'nt')
    @mock.patch.object(processutils.subprocess, 'Popen')
    def test_prlimit_windows(self, mock_popen):
        prlimit = self.limit_address_space()
        mock_popen.return_value.communicate.return_value = None
        processutils.execute(*self.SIMPLE_PROGRAM, prlimit=prlimit, check_exit_code=False)
        mock_popen.assert_called_once_with(self.SIMPLE_PROGRAM, stdin=mock.ANY, stdout=mock.ANY, stderr=mock.ANY, close_fds=mock.ANY, preexec_fn=mock.ANY, shell=mock.ANY, cwd=mock.ANY, env=mock.ANY)

    @mock.patch.object(processutils.subprocess, 'Popen')
    def test_python_exec(self, sub_mock):
        mock_subprocess = mock.MagicMock()
        mock_subprocess.communicate.return_value = (b'', b'')
        sub_mock.return_value = mock_subprocess
        args = ['/a/command']
        prlimit = self.limit_address_space()
        processutils.execute(*args, prlimit=prlimit, check_exit_code=False, python_exec='/fake_path')
        python_path = sub_mock.mock_calls[0][1][0][0]
        self.assertEqual('/fake_path', python_path)