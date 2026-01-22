import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
@property
def VSInstallDir(self):
    """
        Microsoft Visual Studio directory.

        Return
        ------
        str
            path
        """
    default = join(self.ProgramFilesx86, 'Microsoft Visual Studio %0.1f' % self.vs_ver)
    return self.ri.lookup(self.ri.vs, '%0.1f' % self.vs_ver) or default