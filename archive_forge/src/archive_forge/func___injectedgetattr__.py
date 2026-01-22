import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
def __injectedgetattr__(self, name):
    """
    A getattr method to cause a class to be refreshed.
    """
    if name == '__del__':
        raise AttributeError('Without this, Python segfaults.')
    updateInstance(self)
    log.msg(f'(rebuilding stale {reflect.qual(self.__class__)} instance ({name}))')
    result = getattr(self, name)
    return result