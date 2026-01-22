import contextlib
import errno
import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import fasteners
from oslo_config import cfg
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def _lock_wrapper(argv):
    """Create a dir for locks and pass it to command from arguments

    This is exposed as a console script entry point named
    lockutils-wrapper

    If you run this:
        lockutils-wrapper stestr run <etc>

    a temporary directory will be created for all your locks and passed to all
    your tests in an environment variable. The temporary dir will be deleted
    afterwards and the return value will be preserved.
    """
    lock_dir = tempfile.mkdtemp()
    os.environ['OSLO_LOCK_PATH'] = lock_dir
    try:
        ret_val = subprocess.call(argv[1:])
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)
    return ret_val