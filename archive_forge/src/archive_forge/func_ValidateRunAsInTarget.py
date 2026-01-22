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
def ValidateRunAsInTarget(target, target_dict, build_file):
    target_name = target_dict.get('target_name')
    run_as = target_dict.get('run_as')
    if not run_as:
        return
    if type(run_as) is not dict:
        raise GypError("The 'run_as' in target %s from file %s should be a dictionary." % (target_name, build_file))
    action = run_as.get('action')
    if not action:
        raise GypError("The 'run_as' in target %s from file %s must have an 'action' section." % (target_name, build_file))
    if type(action) is not list:
        raise GypError("The 'action' for 'run_as' in target %s from file %s must be a list." % (target_name, build_file))
    working_directory = run_as.get('working_directory')
    if working_directory and type(working_directory) is not str:
        raise GypError("The 'working_directory' for 'run_as' in target %s in file %s should be a string." % (target_name, build_file))
    environment = run_as.get('environment')
    if environment and type(environment) is not dict:
        raise GypError("The 'environment' for 'run_as' in target %s in file %s should be a dictionary." % (target_name, build_file))