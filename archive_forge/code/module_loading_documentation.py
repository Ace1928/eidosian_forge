import copy
import os
import sys
from importlib import import_module
from importlib.util import find_spec as importlib_find

    Find the name of the directory that contains a module, if possible.

    Raise ValueError otherwise, e.g. for namespace packages that are split
    over several directories.
    