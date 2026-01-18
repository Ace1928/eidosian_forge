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

        Given an environment variable name and specified paths,
        return a pathsep-separated string of paths containing
        unique, extant, directories from those paths and from
        the environment variable. Raise an error if no paths
        are resolved.

        Parameters
        ----------
        name: str
            Environment variable name
        spec_path_lists: list of str
            Paths
        exists: bool
            It True, only return existing paths.

        Return
        ------
        str
            Pathsep-separated paths
        