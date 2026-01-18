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
def vs(self):
    """
        Microsoft Visual Studio VS7 registry key.

        Return
        ------
        str
            Registry key
        """
    return join(self.sxs, 'VS7')