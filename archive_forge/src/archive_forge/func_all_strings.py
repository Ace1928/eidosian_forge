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
def all_strings(lst):
    """Return True if all items in lst are string objects. """
    for item in lst:
        if not is_string(item):
            return False
    return True