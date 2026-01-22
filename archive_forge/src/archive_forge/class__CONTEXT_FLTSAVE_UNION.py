from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
class _CONTEXT_FLTSAVE_UNION(Union):
    _fields_ = [('flt', XMM_SAVE_AREA32), ('xmm', _CONTEXT_FLTSAVE_STRUCT)]

    def from_dict(self):
        raise NotImplementedError()

    def to_dict(self):
        d = dict()
        d['flt'] = self.flt.to_dict()
        d['xmm'] = self.xmm.to_dict()
        return d