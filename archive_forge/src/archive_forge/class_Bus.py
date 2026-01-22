import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
class Bus(object):
    """Process state-machine and messenger for HTTP site deployment.

    All listeners for a given channel are guaranteed to be called even
    if others at the same channel fail. Each failure is logged, but
    execution proceeds on to the next listener. The only way to stop all
    processing from inside a listener is to raise SystemExit and stop the
    whole server.
    """
    states = states
    state = states.STOPPED
    execv = False
    max_cloexec_files = max_files

    def __init__(self):
        """Initialize pub/sub bus."""
        self.execv = False
        self.state = states.STOPPED
        channels = ('start', 'stop', 'exit', 'graceful', 'log', 'main')
        self.listeners = dict(((channel, set()) for channel in channels))
        self._priorities = {}

    def subscribe(self, channel, callback=None, priority=None):
        """Add the given callback at the given channel (if not present).

        If callback is None, return a partial suitable for decorating
        the callback.
        """
        if callback is None:
            return functools.partial(self.subscribe, channel, priority=priority)
        ch_listeners = self.listeners.setdefault(channel, set())
        ch_listeners.add(callback)
        if priority is None:
            priority = getattr(callback, 'priority', 50)
        self._priorities[channel, callback] = priority

    def unsubscribe(self, channel, callback):
        """Discard the given callback (if present)."""
        listeners = self.listeners.get(channel)
        if listeners and callback in listeners:
            listeners.discard(callback)
            del self._priorities[channel, callback]

    def publish(self, channel, *args, **kwargs):
        """Return output of all subscribers for the given channel."""
        if channel not in self.listeners:
            return []
        exc = ChannelFailures()
        output = []
        raw_items = ((self._priorities[channel, listener], listener) for listener in self.listeners[channel])
        items = sorted(raw_items, key=operator.itemgetter(0))
        for priority, listener in items:
            try:
                output.append(listener(*args, **kwargs))
            except KeyboardInterrupt:
                raise
            except SystemExit:
                e = sys.exc_info()[1]
                if exc and e.code == 0:
                    e.code = 1
                raise
            except Exception:
                exc.handle_exception()
                if channel == 'log':
                    pass
                else:
                    self.log('Error in %r listener %r' % (channel, listener), level=40, traceback=True)
        if exc:
            raise exc
        return output

    def _clean_exit(self):
        """Assert that the Bus is not running in atexit handler callback."""
        if self.state != states.EXITING:
            warnings.warn('The main thread is exiting, but the Bus is in the %r state; shutting it down automatically now. You must either call bus.block() after start(), or call bus.exit() before the main thread exits.' % self.state, RuntimeWarning)
            self.exit()

    def start(self):
        """Start all services."""
        atexit.register(self._clean_exit)
        self.state = states.STARTING
        self.log('Bus STARTING')
        try:
            self.publish('start')
            self.state = states.STARTED
            self.log('Bus STARTED')
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.log('Shutting down due to error in start listener:', level=40, traceback=True)
            e_info = sys.exc_info()[1]
            try:
                self.exit()
            except Exception:
                pass
            raise e_info

    def exit(self):
        """Stop all services and prepare to exit the process."""
        exitstate = self.state
        EX_SOFTWARE = 70
        try:
            self.stop()
            self.state = states.EXITING
            self.log('Bus EXITING')
            self.publish('exit')
            self.log('Bus EXITED')
        except Exception:
            os._exit(EX_SOFTWARE)
        if exitstate == states.STARTING:
            os._exit(EX_SOFTWARE)

    def restart(self):
        """Restart the process (may close connections).

        This method does not restart the process from the calling thread;
        instead, it stops the bus and asks the main thread to call execv.
        """
        self.execv = True
        self.exit()

    def graceful(self):
        """Advise all services to reload."""
        self.log('Bus graceful')
        self.publish('graceful')

    def block(self, interval=0.1):
        """Wait for the EXITING state, KeyboardInterrupt or SystemExit.

        This function is intended to be called only by the main thread.
        After waiting for the EXITING state, it also waits for all threads
        to terminate, and then calls os.execv if self.execv is True. This
        design allows another thread to call bus.restart, yet have the main
        thread perform the actual execv call (required on some platforms).
        """
        try:
            self.wait(states.EXITING, interval=interval, channel='main')
        except (KeyboardInterrupt, IOError):
            self.log('Keyboard Interrupt: shutting down bus')
            self.exit()
        except SystemExit:
            self.log('SystemExit raised: shutting down bus')
            self.exit()
            raise
        self.log('Waiting for child threads to terminate...')
        for t in threading.enumerate():
            if t != threading.current_thread() and (not isinstance(t, threading._MainThread)) and (not t.daemon):
                self.log('Waiting for thread %s.' % t.name)
                t.join()
        if self.execv:
            self._do_execv()

    def wait(self, state, interval=0.1, channel=None):
        """Poll for the given state(s) at intervals; publish to channel."""
        states = set(always_iterable(state))
        while self.state not in states:
            time.sleep(interval)
            self.publish(channel)

    def _do_execv(self):
        """Re-execute the current process.

        This must be called from the main thread, because certain platforms
        (OS X) don't allow execv to be called in a child thread very well.
        """
        try:
            args = self._get_true_argv()
        except NotImplementedError:
            "It's probably win32 or GAE"
            args = [sys.executable] + self._get_interpreter_argv() + sys.argv
        self.log('Re-spawning %s' % ' '.join(args))
        self._extend_pythonpath(os.environ)
        if sys.platform[:4] == 'java':
            from _systemrestart import SystemRestart
            raise SystemRestart
        else:
            if sys.platform == 'win32':
                args = ['"%s"' % arg for arg in args]
            os.chdir(_startup_cwd)
            if self.max_cloexec_files:
                self._set_cloexec()
            os.execv(sys.executable, args)

    @staticmethod
    def _get_interpreter_argv():
        """Retrieve current Python interpreter's arguments.

        Returns empty tuple in case of frozen mode, uses built-in arguments
        reproduction function otherwise.

        Frozen mode is possible for the app has been packaged into a binary
        executable using py2exe. In this case the interpreter's arguments are
        already built-in into that executable.

        :seealso: https://github.com/cherrypy/cherrypy/issues/1526
        Ref: https://pythonhosted.org/PyInstaller/runtime-information.html
        """
        return [] if getattr(sys, 'frozen', False) else subprocess._args_from_interpreter_flags()

    @staticmethod
    def _get_true_argv():
        """Retrieve all real arguments of the python interpreter.

        ...even those not listed in ``sys.argv``

        :seealso: http://stackoverflow.com/a/28338254/595220
        :seealso: http://stackoverflow.com/a/6683222/595220
        :seealso: http://stackoverflow.com/a/28414807/595220
        """
        try:
            char_p = ctypes.c_wchar_p
            argv = ctypes.POINTER(char_p)()
            argc = ctypes.c_int()
            ctypes.pythonapi.Py_GetArgcArgv(ctypes.byref(argc), ctypes.byref(argv))
            _argv = argv[:argc.value]
            argv_len, is_command, is_module = (len(_argv), False, False)
            try:
                m_ind = _argv.index('-m')
                if m_ind < argv_len - 1 and _argv[m_ind + 1] in ('-c', '-m'):
                    "\n                    In some older Python versions `-m`'s argument may be\n                    substituted with `-c`, not `-m`\n                    "
                    is_module = True
            except (IndexError, ValueError):
                m_ind = None
            try:
                c_ind = _argv.index('-c')
                if c_ind < argv_len - 1 and _argv[c_ind + 1] == '-c':
                    is_command = True
            except (IndexError, ValueError):
                c_ind = None
            if is_module:
                "It's containing `-m -m` sequence of arguments"
                if is_command and c_ind < m_ind:
                    "There's `-c -c` before `-m`"
                    raise RuntimeError("Cannot reconstruct command from '-c'. Ref: https://github.com/cherrypy/cherrypy/issues/1545")
                original_module = sys.argv[0]
                if not os.access(original_module, os.R_OK):
                    "There's no such module exist"
                    raise AttributeError("{} doesn't seem to be a module accessible by current user".format(original_module))
                del _argv[m_ind:m_ind + 2]
                _argv.insert(m_ind, original_module)
            elif is_command:
                "It's containing just `-c -c` sequence of arguments"
                raise RuntimeError("Cannot reconstruct command from '-c'. Ref: https://github.com/cherrypy/cherrypy/issues/1545")
        except AttributeError:
            "It looks Py_GetArgcArgv's completely absent in some environments\n\n            It is known, that there's no Py_GetArgcArgv in MS Windows and\n            ``ctypes`` module is completely absent in Google AppEngine\n\n            :seealso: https://github.com/cherrypy/cherrypy/issues/1506\n            :seealso: https://github.com/cherrypy/cherrypy/issues/1512\n            :ref: http://bit.ly/2gK6bXK\n            "
            raise NotImplementedError
        else:
            return _argv

    @staticmethod
    def _extend_pythonpath(env):
        """Prepend current working dir to PATH environment variable if needed.

        If sys.path[0] is an empty string, the interpreter was likely
        invoked with -m and the effective path is about to change on
        re-exec.  Add the current directory to $PYTHONPATH to ensure
        that the new process sees the same path.

        This issue cannot be addressed in the general case because
        Python cannot reliably reconstruct the
        original command line (http://bugs.python.org/issue14208).

        (This idea filched from tornado.autoreload)
        """
        path_prefix = '.' + os.pathsep
        existing_path = env.get('PYTHONPATH', '')
        needs_patch = sys.path[0] == '' and (not existing_path.startswith(path_prefix))
        if needs_patch:
            env['PYTHONPATH'] = path_prefix + existing_path

    def _set_cloexec(self):
        """Set the CLOEXEC flag on all open files (except stdin/out/err).

        If self.max_cloexec_files is an integer (the default), then on
        platforms which support it, it represents the max open files setting
        for the operating system. This function will be called just before
        the process is restarted via os.execv() to prevent open files
        from persisting into the new process.

        Set self.max_cloexec_files to 0 to disable this behavior.
        """
        for fd in range(3, self.max_cloexec_files):
            try:
                flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            except IOError:
                continue
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def stop(self):
        """Stop all services."""
        self.state = states.STOPPING
        self.log('Bus STOPPING')
        self.publish('stop')
        self.state = states.STOPPED
        self.log('Bus STOPPED')

    def start_with_callback(self, func, args=None, kwargs=None):
        """Start 'func' in a new thread T, then start self (and return T)."""
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        args = (func,) + args

        def _callback(func, *a, **kw):
            self.wait(states.STARTED)
            func(*a, **kw)
        t = threading.Thread(target=_callback, args=args, kwargs=kwargs)
        t.name = 'Bus Callback ' + t.name
        t.start()
        self.start()
        return t

    def log(self, msg='', level=20, traceback=False):
        """Log the given message. Append the last traceback if requested."""
        if traceback:
            msg += '\n' + ''.join(_traceback.format_exception(*sys.exc_info()))
        self.publish('log', msg, level)