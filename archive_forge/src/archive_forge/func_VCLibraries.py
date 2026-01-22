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
def VCLibraries(self):
    """
        Microsoft Visual C++ & Microsoft Foundation Class Libraries.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver >= 15.0:
        arch_subdir = self.pi.target_dir(x64=True)
    else:
        arch_subdir = self.pi.target_dir(hidex86=True)
    paths = ['Lib%s' % arch_subdir, 'ATLMFC\\Lib%s' % arch_subdir]
    if self.vs_ver >= 14.0:
        paths += ['Lib\\store%s' % arch_subdir]
    return [join(self.si.VCInstallDir, path) for path in paths]