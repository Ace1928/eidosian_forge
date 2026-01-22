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
def LoadAutomaticVariablesFromDict(variables, the_dict):
    for key, value in the_dict.items():
        if type(value) in (str, int, list):
            variables['_' + key] = value