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
def _find_latest_available_vs_ver(self):
    """
        Find the latest VC version

        Return
        ------
        float
            version
        """
    reg_vc_vers = self.find_reg_vs_vers()
    if not (reg_vc_vers or self.known_vs_paths):
        raise distutils.errors.DistutilsPlatformError('No Microsoft Visual C++ version found')
    vc_vers = set(reg_vc_vers)
    vc_vers.update(self.known_vs_paths)
    return sorted(vc_vers)[-1]