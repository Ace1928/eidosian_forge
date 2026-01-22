import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _check_and_log_subprocess(command, logger, **kwargs):
    """
    Run *command*, returning its stdout output if it succeeds.

    If it fails (exits with nonzero return code), raise an exception whose text
    includes the failed command and captured stdout and stderr output.

    Regardless of the return code, the command is logged at DEBUG level on
    *logger*.  In case of success, the output is likewise logged.
    """
    logger.debug('%s', _pformat_subprocess(command))
    proc = subprocess.run(command, capture_output=True, **kwargs)
    if proc.returncode:
        stdout = proc.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        stderr = proc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        raise RuntimeError(f'The command\n    {_pformat_subprocess(command)}\nfailed and generated the following output:\n{stdout}\nand the following error:\n{stderr}')
    if proc.stdout:
        logger.debug('stdout:\n%s', proc.stdout)
    if proc.stderr:
        logger.debug('stderr:\n%s', proc.stderr)
    return proc.stdout