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
@staticmethod
def _use_last_dir_name(path, prefix=''):
    """
        Return name of the last dir in path or '' if no dir found.

        Parameters
        ----------
        path: str
            Use dirs in this path
        prefix: str
            Use only dirs starting by this prefix

        Return
        ------
        str
            name
        """
    matching_dirs = (dir_name for dir_name in reversed(listdir(path)) if isdir(join(path, dir_name)) and dir_name.startswith(prefix))
    return next(matching_dirs, None) or ''