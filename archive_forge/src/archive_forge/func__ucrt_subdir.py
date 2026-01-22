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
def _ucrt_subdir(self):
    """
        Microsoft Universal C Runtime SDK version subdir.

        Return
        ------
        str
            subdir
        """
    ucrtver = self.si.UniversalCRTSdkLastVersion
    return '%s\\' % ucrtver if ucrtver else ''