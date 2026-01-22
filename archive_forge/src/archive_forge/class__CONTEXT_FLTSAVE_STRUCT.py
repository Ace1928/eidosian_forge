from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
class _CONTEXT_FLTSAVE_STRUCT(Structure):
    _fields_ = [('Header', M128A * 2), ('Legacy', M128A * 8), ('Xmm0', M128A), ('Xmm1', M128A), ('Xmm2', M128A), ('Xmm3', M128A), ('Xmm4', M128A), ('Xmm5', M128A), ('Xmm6', M128A), ('Xmm7', M128A), ('Xmm8', M128A), ('Xmm9', M128A), ('Xmm10', M128A), ('Xmm11', M128A), ('Xmm12', M128A), ('Xmm13', M128A), ('Xmm14', M128A), ('Xmm15', M128A)]

    def from_dict(self):
        raise NotImplementedError()

    def to_dict(self):
        d = dict()
        for name, type in self._fields_:
            if name in ('Header', 'Legacy'):
                d[name] = tuple([x.Low + (x.High << 64) for x in getattr(self, name)])
            else:
                x = getattr(self, name)
                d[name] = x.Low + (x.High << 64)
        return d