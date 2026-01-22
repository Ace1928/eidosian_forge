import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def TurnIntIntoStrInDict(the_dict):
    """Given dict the_dict, recursively converts all integers into strings.
  """
    for k, v in the_dict.items():
        if type(v) is int:
            v = str(v)
            the_dict[k] = v
        elif type(v) is dict:
            TurnIntIntoStrInDict(v)
        elif type(v) is list:
            TurnIntIntoStrInList(v)
        if type(k) is int:
            del the_dict[k]
            the_dict[str(k)] = v