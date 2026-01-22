import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def add_library(self, name, sources, **build_info):
    """
        Add library to configuration.

        Parameters
        ----------
        name : str
            Name of the extension.
        sources : sequence
            List of the sources. The list of sources may contain functions
            (called source generators) which must take an extension instance
            and a build directory as inputs and return a source file or list of
            source files or None. If None is returned then no sources are
            generated. If the Extension instance has no sources after
            processing all source generators, then no extension module is
            built.
        build_info : dict, optional
            The following keys are allowed:

                * depends
                * macros
                * include_dirs
                * extra_compiler_args
                * extra_f77_compile_args
                * extra_f90_compile_args
                * f2py_options
                * language

        """
    self._add_library(name, sources, None, build_info)
    dist = self.get_distribution()
    if dist is not None:
        self.warn('distutils distribution has been initialized, it may be too late to add a library ' + name)