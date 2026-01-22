import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
class BaseProcess(object):
    """
    Process objects represent activity that is run in a separate process

    The class is analogous to `threading.Thread`
    """

    def _Popen(self):
        raise NotImplementedError

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        assert group is None, 'group argument must be None for now'
        count = next(_process_counter)
        self._identity = _current_process._identity + (count,)
        self._config = _current_process._config.copy()
        self._parent_pid = os.getpid()
        self._parent_name = _current_process.name
        self._popen = None
        self._closed = False
        self._target = target
        self._args = tuple(args)
        self._kwargs = dict(kwargs)
        self._name = name or type(self).__name__ + '-' + ':'.join((str(i) for i in self._identity))
        if daemon is not None:
            self.daemon = daemon
        _dangling.add(self)

    def _check_closed(self):
        if self._closed:
            raise ValueError('process object is closed')

    def run(self):
        """
        Method to be run in sub-process; can be overridden in sub-class
        """
        if self._target:
            self._target(*self._args, **self._kwargs)

    def start(self):
        """
        Start child process
        """
        self._check_closed()
        assert self._popen is None, 'cannot start a process twice'
        assert self._parent_pid == os.getpid(), 'can only start a process object created by current process'
        assert not _current_process._config.get('daemon'), 'daemonic processes are not allowed to have children'
        _cleanup()
        self._popen = self._Popen(self)
        self._sentinel = self._popen.sentinel
        del self._target, self._args, self._kwargs
        _children.add(self)

    def terminate(self):
        """
        Terminate process; sends SIGTERM signal or uses TerminateProcess()
        """
        self._check_closed()
        self._popen.terminate()

    def kill(self):
        """
        Terminate process; sends SIGKILL signal or uses TerminateProcess()
        """
        self._check_closed()
        self._popen.kill()

    def join(self, timeout=None):
        """
        Wait until child process terminates
        """
        self._check_closed()
        assert self._parent_pid == os.getpid(), 'can only join a child process'
        assert self._popen is not None, 'can only join a started process'
        res = self._popen.wait(timeout)
        if res is not None:
            _children.discard(self)

    def is_alive(self):
        """
        Return whether process is alive
        """
        self._check_closed()
        if self is _current_process:
            return True
        assert self._parent_pid == os.getpid(), 'can only test a child process'
        if self._popen is None:
            return False
        returncode = self._popen.poll()
        if returncode is None:
            return True
        else:
            _children.discard(self)
            return False

    def close(self):
        """
        Close the Process object.

        This method releases resources held by the Process object.  It is
        an error to call this method if the child process is still running.
        """
        if self._popen is not None:
            if self._popen.poll() is None:
                raise ValueError('Cannot close a process while it is still running. You should first call join() or terminate().')
            self._popen.close()
            self._popen = None
            del self._sentinel
            _children.discard(self)
        self._closed = True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), 'name must be a string'
        self._name = name

    @property
    def daemon(self):
        """
        Return whether process is a daemon
        """
        return self._config.get('daemon', False)

    @daemon.setter
    def daemon(self, daemonic):
        """
        Set whether process is a daemon
        """
        assert self._popen is None, 'process has already started'
        self._config['daemon'] = daemonic

    @property
    def authkey(self):
        return self._config['authkey']

    @authkey.setter
    def authkey(self, authkey):
        """
        Set authorization key of process
        """
        self._config['authkey'] = AuthenticationString(authkey)

    @property
    def exitcode(self):
        """
        Return exit code of process or `None` if it has yet to stop
        """
        self._check_closed()
        if self._popen is None:
            return self._popen
        return self._popen.poll()

    @property
    def ident(self):
        """
        Return identifier (PID) of process or `None` if it has yet to start
        """
        self._check_closed()
        if self is _current_process:
            return os.getpid()
        else:
            return self._popen and self._popen.pid
    pid = ident

    @property
    def sentinel(self):
        """
        Return a file descriptor (Unix) or handle (Windows) suitable for
        waiting for process termination.
        """
        self._check_closed()
        try:
            return self._sentinel
        except AttributeError:
            raise ValueError('process not started') from None

    def __repr__(self):
        exitcode = None
        if self is _current_process:
            status = 'started'
        elif self._closed:
            status = 'closed'
        elif self._parent_pid != os.getpid():
            status = 'unknown'
        elif self._popen is None:
            status = 'initial'
        else:
            exitcode = self._popen.poll()
            if exitcode is not None:
                status = 'stopped'
            else:
                status = 'started'
        info = [type(self).__name__, 'name=%r' % self._name]
        if self._popen is not None:
            info.append('pid=%s' % self._popen.pid)
        info.append('parent=%s' % self._parent_pid)
        info.append(status)
        if exitcode is not None:
            exitcode = _exitcode_to_name.get(exitcode, exitcode)
            info.append('exitcode=%s' % exitcode)
        if self.daemon:
            info.append('daemon')
        return '<%s>' % ' '.join(info)

    def _bootstrap(self, parent_sentinel=None):
        from . import util, context
        global _current_process, _parent_process, _process_counter, _children
        try:
            if self._start_method is not None:
                context._force_start_method(self._start_method)
            _process_counter = itertools.count(1)
            _children = set()
            util._close_stdin()
            old_process = _current_process
            _current_process = self
            _parent_process = _ParentProcess(self._parent_name, self._parent_pid, parent_sentinel)
            if threading._HAVE_THREAD_NATIVE_ID:
                threading.main_thread()._set_native_id()
            try:
                self._after_fork()
            finally:
                del old_process
            util.info('child process calling self.run()')
            try:
                self.run()
                exitcode = 0
            finally:
                util._exit_function()
        except SystemExit as e:
            if e.code is None:
                exitcode = 0
            elif isinstance(e.code, int):
                exitcode = e.code
            else:
                sys.stderr.write(str(e.code) + '\n')
                exitcode = 1
        except:
            exitcode = 1
            import traceback
            sys.stderr.write('Process %s:\n' % self.name)
            traceback.print_exc()
        finally:
            threading._shutdown()
            util.info('process exiting with exitcode %d' % exitcode)
            util._flush_std_streams()
        return exitcode

    @staticmethod
    def _after_fork():
        from . import util
        util._finalizer_registry.clear()
        util._run_after_forkers()