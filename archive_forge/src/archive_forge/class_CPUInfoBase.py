import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
class CPUInfoBase:
    """Holds CPU information and provides methods for requiring
    the availability of various CPU features.
    """

    def _try_call(self, func):
        try:
            return func()
        except Exception:
            pass

    def __getattr__(self, name):
        if not name.startswith('_'):
            if hasattr(self, '_' + name):
                attr = getattr(self, '_' + name)
                if isinstance(attr, types.MethodType):
                    return lambda func=self._try_call, attr=attr: func(attr)
            else:
                return lambda: None
        raise AttributeError(name)

    def _getNCPUs(self):
        return 1

    def __get_nbits(self):
        abits = platform.architecture()[0]
        nbits = re.compile('(\\d+)bit').search(abits).group(1)
        return nbits

    def _is_32bit(self):
        return self.__get_nbits() == '32'

    def _is_64bit(self):
        return self.__get_nbits() == '64'