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
def VCStoreRefs(self):
    """
        Microsoft Visual C++ store references Libraries.

        Return
        ------
        list of str
            paths
        """
    if self.vs_ver < 14.0:
        return []
    return [join(self.si.VCInstallDir, 'Lib\\store\\references')]