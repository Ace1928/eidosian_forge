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
def WindowsSdkVersion(self):
    """
        Microsoft Windows SDK versions for specified MSVC++ version.

        Return
        ------
        tuple of str
            versions
        """
    if self.vs_ver <= 9.0:
        return ('7.0', '6.1', '6.0a')
    elif self.vs_ver == 10.0:
        return ('7.1', '7.0a')
    elif self.vs_ver == 11.0:
        return ('8.0', '8.0a')
    elif self.vs_ver == 12.0:
        return ('8.1', '8.1a')
    elif self.vs_ver >= 14.0:
        return ('10.0', '8.1')
    return None