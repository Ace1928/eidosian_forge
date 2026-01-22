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
def NetFxSdkVersion(self):
    """
        Microsoft .NET Framework SDK versions.

        Return
        ------
        tuple of str
            versions
        """
    return ('4.7.2', '4.7.1', '4.7', '4.6.2', '4.6.1', '4.6', '4.5.2', '4.5.1', '4.5') if self.vs_ver >= 14.0 else ()