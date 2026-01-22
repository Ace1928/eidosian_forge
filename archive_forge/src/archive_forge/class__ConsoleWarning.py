from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
class _ConsoleWarning(object):

    def __init__(self, verbose=1):
        if verbose is True:
            verbose = 1
        elif verbose is False:
            verbose = 0
        self.verbose = verbose

    @staticmethod
    def warning(s: str) -> None:
        _console_warning(s, rpy2.rinterface_lib.callbacks.logger.warning)

    @staticmethod
    def debug(s: str) -> None:
        _console_warning(s, rpy2.rinterface_lib.callbacks.logger.debug)

    @staticmethod
    def set(fun):
        if not hasattr(_ConsoleWarning, 'builtin_warning'):
            _ConsoleWarning.builtin_warning = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = fun

    @staticmethod
    def set_warning():
        _ConsoleWarning.set(_ConsoleWarning.warning)

    @staticmethod
    def set_debug():
        _ConsoleWarning.set(_ConsoleWarning.debug)

    @staticmethod
    def set_builtin():
        _ConsoleWarning.set(_ConsoleWarning.builtin_warning)

    def __enter__(self):
        self.previous_warning = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        if self.verbose > 0:
            self.set_warning()
        else:
            self.set_debug()

    def __exit__(self, type, value, traceback):
        self.set(self.previous_warning)