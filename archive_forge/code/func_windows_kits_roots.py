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
def windows_kits_roots(self):
    """
        Microsoft Windows Kits Roots registry key.

        Return
        ------
        str
            Registry key
        """
    return 'Windows Kits\\Installed Roots'