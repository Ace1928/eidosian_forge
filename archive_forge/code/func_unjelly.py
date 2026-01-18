import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def unjelly(self, ao):
    try:
        l = [None]
        self.unjellyInto(l, 0, ao)
        for func, v in self.afterUnjelly:
            func(v[0])
        return l[0]
    except BaseException:
        log.msg('Error jellying object! Stacktrace follows::')
        log.msg('\n'.join(map(repr, self.stack)))
        raise