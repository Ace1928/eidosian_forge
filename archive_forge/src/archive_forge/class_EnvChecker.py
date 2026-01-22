import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
class EnvChecker:
    is_colab = _colab
    is_notebook = _notebook
    is_jpy = _notebook
    cpu_cores = _CPU_CORES
    max_threads = _MAX_THREADS
    max_procs = _MAX_PROCS
    loggers: Optional[Dict[str, ThreadSafeHandler]] = {}
    handlers: Optional[Dict[str, Any]] = {}
    watcher_enabled: Optional[bool] = False
    threads: Optional[List[threading.Thread]] = []
    sigs: Optional[Dict[str, signal.signal]] = {}

    @classmethod
    def get_logger(cls, name='LazyOps', module=None, *args, **kwargs):
        if not EnvChecker.loggers.get(name):
            EnvChecker.loggers[name] = ThreadSafeHandler()
        if module:
            return EnvChecker.loggers[name].get(setup_new_logger, *args, name=name, **kwargs).get_logger(module=module)
        return EnvChecker.loggers[name].get(setup_new_logger, *args, name=name, **kwargs)

    @property
    def alive(self):
        return IsLazyAlive

    @property
    def killed(self):
        return not IsLazyAlive

    @property
    def is_threadsafe(self):
        return bool(threading.current_thread() is threading.main_thread())

    @classmethod
    def set_state(cls, state: bool):
        global IsLazyAlive
        IsLazyAlive = state
    set_alive = partialmethod(set_state, True)
    set_dead = partialmethod(set_state, False)

    @classmethod
    def exit_handler(cls, signum, frame):
        logger = EnvChecker.loggers['LazyWatch']
        logger.error('Received SIGINT or SIGTERM! Gracefully Exiting.')
        if EnvChecker.handlers:
            for handler, func in EnvChecker.handlers.items():
                logger.error(f'Calling Exit for {handler}')
                func()
        if EnvChecker.threads:
            for thread in EnvChecker.threads:
                thread.join()
        EnvChecker.set_dead()
        sys.exit(0)

    @classmethod
    def enable_watcher(cls):
        if EnvChecker.watcher_enabled:
            return
        EnvChecker.sigs['sigint'] = signal.signal(signal.SIGINT, EnvChecker.exit_handler)
        EnvChecker.sigs['sigterm'] = signal.signal(signal.SIGTERM, EnvChecker.exit_handler)
        EnvChecker.loggers['LazyWatch'] = EnvChecker.get_logger(name='LazyWatch')
        EnvChecker.watcher_enabled = True

    @classmethod
    def add_thread(cls, t):
        EnvChecker.threads.append(t)

    @classmethod
    def set_multiparams(cls, max_procs: int=None, max_threads: int=None):
        if max_procs is not None:
            EnvChecker.max_procs = max_procs
        if max_threads is not None:
            EnvChecker.max_threads = max_threads

    @classmethod
    def add_exit_handler(cls, name, func):
        if name not in EnvChecker.handlers:
            EnvChecker.handlers[name] = func

    @classmethod
    def getset(cls, name, val=None, default=None, set_if_none=False):
        eval = os.environ.get(name, default=default)
        if not eval and set_if_none or (eval and eval != val):
            os.environ[name] = str(val)
            return os.environ.get(name)
        if not val or not set_if_none:
            return eval
        return eval

    def __call__(self, name, val=None, default=None, set_if_none=None, *args, **kwargs):
        return EnvChecker.getset(name, *args, val=None, default=None, set_if_none=None, **kwargs)

    def __exit__(self, type, value, traceback, *args, **kwargs):
        if EnvChecker.watcher_enabled:
            if self.killed:
                sys.exit(0)
            signal.signal(signal.SIGINT, EnvChecker.sigs['sigint'])
            signal.signal(signal.SIGTERM, EnvChecker.sigs['sigterm'])
        EnvChecker.set_dead()

    def watch(self):
        self.enable_watcher()
        return self