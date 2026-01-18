from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
@staticmethod
def warning(s: str) -> None:
    _console_warning(s, rpy2.rinterface_lib.callbacks.logger.warning)