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
def LoadVariablesFromVariablesDict(variables, the_dict, the_dict_key):
    for key, value in the_dict.get('variables', {}).items():
        if type(value) not in (str, int, list):
            continue
        if key.endswith('%'):
            variable_name = key[:-1]
            if variable_name in variables:
                continue
            if the_dict_key == 'variables' and variable_name in the_dict:
                value = the_dict[variable_name]
        else:
            variable_name = key
        variables[variable_name] = value