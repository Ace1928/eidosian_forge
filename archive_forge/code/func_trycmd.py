import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def trycmd(*args, **kwargs):
    """A wrapper around execute() to more easily handle warnings and errors.

    Returns an (out, err) tuple of strings containing the output of
    the command's stdout and stderr.  If 'err' is not empty then the
    command can be considered to have failed.

    :param discard_warnings:  True | False. Defaults to False. If set to True,
                              then for succeeding commands, stderr is cleared
    :type discard_warnings:   boolean
    :returns:                 (out, err) from process execution

    """
    discard_warnings = kwargs.pop('discard_warnings', False)
    try:
        out, err = execute(*args, **kwargs)
        failed = False
    except ProcessExecutionError as exn:
        out, err = ('', str(exn))
        failed = True
    if not failed and discard_warnings and err:
        err = ''
    return (out, err)