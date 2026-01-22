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
def _commandline_dep_string(cc_args, extra_postargs, pp_opts):
    """
    Return commandline representation used to determine if a file needs
    to be recompiled
    """
    cmdline = 'commandline: '
    cmdline += ' '.join(cc_args)
    cmdline += ' '.join(extra_postargs)
    cmdline += ' '.join(pp_opts) + '\n'
    return cmdline