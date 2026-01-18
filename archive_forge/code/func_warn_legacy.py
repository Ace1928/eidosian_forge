import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re
import numpy as np
from ase.formula import formula_hill, formula_metal
def warn_legacy(feature_name):
    warnings.warn(f'The {feature_name} feature is untested and ASE developers do not know whether it works or how to use it.  Please rehabilitate it (by writing unittests) or it may be removed.', FutureWarning)