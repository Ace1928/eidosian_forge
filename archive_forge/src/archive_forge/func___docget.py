import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def __docget(self):
    try:
        return self.__doc
    except AttributeError:
        if '__doc__' in self.__map__:
            return self.__makeattr('__doc__')