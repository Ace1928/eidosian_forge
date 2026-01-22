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
def WindowsSdkLastVersion(self):
    """
        Microsoft Windows SDK last version.

        Return
        ------
        str
            version
        """
    return self._use_last_dir_name(join(self.WindowsSdkDir, 'lib'))