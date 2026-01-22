import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
class ShellOutSSHClient(BaseSSHClient):
    """
    This client shells out to "ssh" binary to run commands on the remote
    server.

    Note: This client should not be used in production.
    """

    def __init__(self, hostname, port=22, username='root', password=None, key=None, key_files=None, timeout=None):
        super().__init__(hostname=hostname, port=port, username=username, password=password, key=key, key_files=key_files, timeout=timeout)
        if self.password:
            raise ValueError('ShellOutSSHClient only supports key auth')
        child = subprocess.Popen(['ssh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        child.communicate()
        if child.returncode == 127:
            raise ValueError('ssh client is not available')
        self.logger = self._get_and_setup_logger()

    def connect(self):
        """
        This client doesn't support persistent connections establish a new
        connection every time "run" method is called.
        """
        return True

    def run(self, cmd, timeout=None):
        return self._run_remote_shell_command([cmd])

    def put(self, path, contents=None, chmod=None, mode='w'):
        if mode == 'w':
            redirect = '>'
        elif mode == 'a':
            redirect = '>>'
        else:
            raise ValueError('Invalid mode: ' + mode)
        cmd = ['echo "{}" {} {}'.format(contents, redirect, path)]
        self._run_remote_shell_command(cmd)
        return path

    def putfo(self, path, fo=None, chmod=None):
        content = fo.read()
        return self.put(path=path, contents=content, chmod=chmod)

    def delete(self, path):
        cmd = ['rm', '-rf', path]
        self._run_remote_shell_command(cmd)
        return True

    def close(self):
        return True

    def _get_base_ssh_command(self):
        cmd = ['ssh']
        if self.key_files:
            self.key_files = cast(str, self.key_files)
            cmd += ['-i', self.key_files]
        if self.timeout:
            cmd += ['-oConnectTimeout=%s' % self.timeout]
        cmd += ['{}@{}'.format(self.username, self.hostname)]
        return cmd

    def _run_remote_shell_command(self, cmd):
        """
        Run a command on a remote server.

        :param      cmd: Command to run.
        :type       cmd: ``list`` of ``str``

        :return: Command stdout, stderr and status code.
        :rtype: ``tuple``
        """
        base_cmd = self._get_base_ssh_command()
        full_cmd = base_cmd + [' '.join(cmd)]
        self.logger.debug('Executing command: "%s"' % ' '.join(full_cmd))
        child = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = child.communicate()
        stdout_str = cast(str, stdout)
        stderr_str = cast(str, stdout)
        return (stdout_str, stderr_str, child.returncode)