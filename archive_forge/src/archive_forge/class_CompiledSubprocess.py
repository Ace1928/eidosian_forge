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
class CompiledSubprocess:
    is_crashed = False

    def __init__(self, executable, env_vars=None):
        self._executable = executable
        self._env_vars = env_vars
        self._inference_state_deletion_queue = collections.deque()
        self._cleanup_callable = lambda: None

    def __repr__(self):
        pid = os.getpid()
        return '<%s _executable=%r, is_crashed=%r, pid=%r>' % (self.__class__.__name__, self._executable, self.is_crashed, pid)

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

    def run(self, inference_state, function, args=(), kwargs={}):
        while True:
            try:
                inference_state_id = self._inference_state_deletion_queue.pop()
            except IndexError:
                break
            else:
                self._send(inference_state_id, None)
        assert callable(function)
        return self._send(id(inference_state), function, args, kwargs)

    def get_sys_path(self):
        return self._send(None, functions.get_sys_path, (), {})

    def _kill(self):
        self.is_crashed = True
        self._cleanup_callable()

    def _send(self, inference_state_id, function, args=(), kwargs={}):
        if self.is_crashed:
            raise InternalError('The subprocess %s has crashed.' % self._executable)
        data = (inference_state_id, function, args, kwargs)
        try:
            pickle_dump(data, self._get_process().stdin, PICKLE_PROTOCOL)
        except BrokenPipeError:
            self._kill()
            raise InternalError('The subprocess %s was killed. Maybe out of memory?' % self._executable)
        try:
            is_exception, traceback, result = pickle_load(self._get_process().stdout)
        except EOFError as eof_error:
            try:
                stderr = self._get_process().stderr.read().decode('utf-8', 'replace')
            except Exception as exc:
                stderr = '<empty/not available (%r)>' % exc
            self._kill()
            _add_stderr_to_debug(self._stderr_queue)
            raise InternalError('The subprocess %s has crashed (%r, stderr=%s).' % (self._executable, eof_error, stderr))
        _add_stderr_to_debug(self._stderr_queue)
        if is_exception:
            result.args = (traceback,)
            raise result
        return result

    def delete_inference_state(self, inference_state_id):
        """
        Currently we are not deleting inference_state instantly. They only get
        deleted once the subprocess is used again. It would probably a better
        solution to move all of this into a thread. However, the memory usage
        of a single inference_state shouldn't be that high.
        """
        self._inference_state_deletion_queue.append(inference_state_id)