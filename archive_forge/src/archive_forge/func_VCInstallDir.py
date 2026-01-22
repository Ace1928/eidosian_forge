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
def VCInstallDir(self):
    """
        Microsoft Visual C++ directory.

        Return
        ------
        str
            path
        """
    path = self._guess_vc() or self._guess_vc_legacy()
    if not isdir(path):
        msg = 'Microsoft Visual C++ directory not found'
        raise distutils.errors.DistutilsPlatformError(msg)
    return path