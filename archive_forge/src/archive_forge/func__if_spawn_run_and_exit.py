import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
def _if_spawn_run_and_exit():
    """If spawned process, run requested spawn task and exit. Else a no-op."""
    is_spawned = '-c' in sys.argv[1:] and sys.argv[sys.argv.index('-c') + 1].startswith('from multiprocessing.')
    if not is_spawned:
        return
    cmd = sys.argv[sys.argv.index('-c') + 1]
    sys.argv = sys.argv[0:1]
    exec(cmd)
    sys.exit(0)