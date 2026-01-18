from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def verify_directory(dir_):
    """create and/or verify a filesystem directory."""
    tries = 0
    while not os.path.exists(dir_):
        try:
            tries += 1
            os.makedirs(dir_, 493)
        except:
            if tries > 5:
                raise