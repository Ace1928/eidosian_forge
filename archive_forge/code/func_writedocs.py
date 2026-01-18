import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def writedocs(dir, pkgpath='', done=None):
    """Write out HTML documentation for all modules in a directory tree."""
    if done is None:
        done = {}
    for importer, modname, ispkg in pkgutil.walk_packages([dir], pkgpath):
        writedoc(modname)
    return