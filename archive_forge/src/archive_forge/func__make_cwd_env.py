import sys
import os
import os.path as op
import tempfile
from subprocess import Popen, check_output, PIPE, STDOUT, CalledProcessError
from srsly.cloudpickle.compat import pickle
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import psutil
from srsly.cloudpickle import dumps
from subprocess import TimeoutExpired
def _make_cwd_env():
    """Helper to prepare environment for the child processes"""
    cloudpickle_repo_folder = op.normpath(op.join(op.dirname(__file__), '..'))
    env = os.environ.copy()
    pythonpath = '{src}{sep}tests{pathsep}{src}'.format(src=cloudpickle_repo_folder, sep=os.sep, pathsep=os.pathsep)
    env['PYTHONPATH'] = pythonpath
    return (cloudpickle_repo_folder, env)