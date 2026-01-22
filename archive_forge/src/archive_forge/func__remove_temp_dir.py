import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _remove_temp_dir(rmtree, tempdir):
    rmtree(tempdir)
    current_process = process.current_process()
    if current_process is not None:
        current_process._config['tempdir'] = None