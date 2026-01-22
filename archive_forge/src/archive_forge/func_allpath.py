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
def allpath(name):
    """Convert a /-separated pathname to one using the OS's path separator."""
    split = name.split('/')
    return os.path.join(*split)