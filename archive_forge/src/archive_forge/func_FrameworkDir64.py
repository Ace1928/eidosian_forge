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
def FrameworkDir64(self):
    """
        Microsoft .NET Framework 64bit directory.

        Return
        ------
        str
            path
        """
    guess_fw = join(self.WinDir, 'Microsoft.NET\\Framework64')
    return self.ri.lookup(self.ri.vc, 'frameworkdir64') or guess_fw