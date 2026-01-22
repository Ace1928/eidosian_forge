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
def add_scripts(self, *files):
    """Add scripts to configuration.

        Add the sequence of files to the beginning of the scripts list.
        Scripts will be installed under the <prefix>/bin/ directory.

        """
    scripts = self.paths(files)
    dist = self.get_distribution()
    if dist is not None:
        if dist.scripts is None:
            dist.scripts = []
        dist.scripts.extend(scripts)
    else:
        self.scripts.extend(scripts)