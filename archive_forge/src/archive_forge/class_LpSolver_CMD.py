import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
class LpSolver_CMD(LpSolver):
    """A generic command line LP Solver"""
    name = 'LpSolver_CMD'

    def __init__(self, path=None, keepFiles=False, *args, **kwargs):
        """

        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param list options: list of additional options to pass to solver (format depends on the solver)
        :param float timeLimit: maximum time for solver (in seconds)
        :param str path: a path to the solver binary
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param args: parameters to pass to :py:class:`LpSolver`
        :param kwargs: parameters to pass to :py:class:`LpSolver`
        """
        LpSolver.__init__(self, *args, **kwargs)
        if path is None:
            self.path = self.defaultPath()
        else:
            self.path = path
        self.keepFiles = keepFiles
        self.setTmpDir()

    def copy(self):
        """Make a copy of self"""
        aCopy = LpSolver.copy(self)
        aCopy.path = self.path
        aCopy.keepFiles = self.keepFiles
        aCopy.tmpDir = self.tmpDir
        return aCopy

    def setTmpDir(self):
        """Set the tmpDir attribute to a reasonnable location for a temporary
        directory"""
        if os.name != 'nt':
            self.tmpDir = os.environ.get('TMPDIR', '/tmp')
            self.tmpDir = os.environ.get('TMP', self.tmpDir)
        else:
            self.tmpDir = os.environ.get('TMPDIR', '')
            self.tmpDir = os.environ.get('TMP', self.tmpDir)
            self.tmpDir = os.environ.get('TEMP', self.tmpDir)
        if not os.path.isdir(self.tmpDir):
            self.tmpDir = ''
        elif not os.access(self.tmpDir, os.F_OK + os.W_OK):
            self.tmpDir = ''

    def create_tmp_files(self, name, *args):
        if self.keepFiles:
            prefix = name
        else:
            prefix = os.path.join(self.tmpDir, uuid4().hex)
        return (f'{prefix}-pulp.{n}' for n in args)

    def silent_remove(self, file: Union[str, bytes, os.PathLike]) -> None:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass

    def delete_tmp_files(self, *args):
        if self.keepFiles:
            return
        for file in args:
            self.silent_remove(file)

    def defaultPath(self):
        raise NotImplementedError

    @staticmethod
    def executableExtension(name):
        if os.name != 'nt':
            return name
        else:
            return name + '.exe'

    @staticmethod
    def executable(command):
        """Checks that the solver command is executable,
        And returns the actual path to it."""
        return shutil.which(command)