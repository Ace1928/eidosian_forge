import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
from jedi.api.exceptions import InternalError
def _GeneralizedPopen(*args, **kwargs):
    if os.name == 'nt':
        try:
            CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
        except AttributeError:
            CREATE_NO_WINDOW = 134217728
        kwargs['creationflags'] = CREATE_NO_WINDOW
    kwargs['close_fds'] = 'posix' in sys.builtin_module_names
    return subprocess.Popen(*args, **kwargs)