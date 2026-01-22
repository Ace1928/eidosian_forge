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
def UCRTLibraries(self):
    """
        Microsoft Universal C Runtime SDK Libraries.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver < 14.0:
        return []
    arch_subdir = self.pi.target_dir(x64=True)
    lib = join(self.si.UniversalCRTSdkDir, 'lib')
    ucrtver = self._ucrt_subdir
    return [join(lib, '%sucrt%s' % (ucrtver, arch_subdir))]