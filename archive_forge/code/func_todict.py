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
def todict(self):
    """
        Return a dictionary compatible with the keyword arguments of distutils
        setup function.

        Examples
        --------
        >>> setup(**config.todict())                           #doctest: +SKIP
        """
    self._optimize_data_files()
    d = {}
    known_keys = self.list_keys + self.dict_keys + self.extra_keys
    for n in known_keys:
        a = getattr(self, n)
        if a:
            d[n] = a
    return d