from __future__ import annotations
import errno
import os
import sys
from typing import Callable
from typing import TYPE_CHECKING
from typing import TypeVar
def _geterrnoclass(self, eno: int) -> type[Error]:
    try:
        return self._errno2class[eno]
    except KeyError:
        clsname = errno.errorcode.get(eno, 'UnknownErrno%d' % (eno,))
        errorcls = type(clsname, (Error,), {'__module__': 'py.error', '__doc__': os.strerror(eno)})
        self._errno2class[eno] = errorcls
        return errorcls