import errno
import logging
import os
import subprocess
import typing
from abc import ABC
from abc import abstractmethod
from io import IOBase
from platform import system
from subprocess import PIPE
from time import sleep
from urllib import request
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException
from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import utils
def _start_process(self, path: str) -> None:
    """Creates a subprocess by executing the command provided.

        :param cmd: full command to execute
        """
    cmd = [path]
    cmd.extend(self.command_line_args())
    close_file_descriptors = self.popen_kw.pop('close_fds', system() != 'Windows')
    try:
        start_info = None
        if system() == 'Windows':
            start_info = subprocess.STARTUPINFO()
            start_info.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
            start_info.wShowWindow = subprocess.SW_HIDE
        self.process = subprocess.Popen(cmd, env=self.env, close_fds=close_file_descriptors, stdout=self.log_output, stderr=self.log_output, stdin=PIPE, creationflags=self.creation_flags, startupinfo=start_info, **self.popen_kw)
        logger.debug('Started executable: `%s` in a child process with pid: %s using %s to output %s', self._path, self.process.pid, self.creation_flags, self.log_output)
    except TypeError:
        raise
    except OSError as err:
        if err.errno == errno.EACCES:
            raise WebDriverException(f"'{os.path.basename(self._path)}' executable may have wrong permissions.") from err
        raise