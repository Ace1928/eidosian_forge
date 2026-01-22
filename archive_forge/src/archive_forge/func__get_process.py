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
@memoize_method
def _get_process(self):
    debug.dbg('Start environment subprocess %s', self._executable)
    parso_path = sys.modules['parso'].__file__
    args = (self._executable, _MAIN_PATH, os.path.dirname(os.path.dirname(parso_path)), '.'.join((str(x) for x in sys.version_info[:3])))
    process = _GeneralizedPopen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self._env_vars)
    self._stderr_queue = queue.Queue()
    self._stderr_thread = t = Thread(target=_enqueue_output, args=(process.stderr, self._stderr_queue))
    t.daemon = True
    t.start()
    self._cleanup_callable = weakref.finalize(self, _cleanup_process, process, t)
    return process