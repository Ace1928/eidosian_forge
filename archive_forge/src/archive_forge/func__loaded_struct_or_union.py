import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_struct_or_union(self, tp):
    if tp.fldnames is None:
        return
    self.ffi._get_cached_btype(tp)
    if tp in self._struct_pending_verification:

        def check(realvalue, expectedvalue, msg):
            if realvalue != expectedvalue:
                raise VerificationError('%s (we have %d, but C compiler says %d)' % (msg, expectedvalue, realvalue))
        ffi = self.ffi
        BStruct = ffi._get_cached_btype(tp)
        layout, cname = self._struct_pending_verification.pop(tp)
        check(layout[0], ffi.sizeof(BStruct), 'wrong total size')
        check(layout[1], ffi.alignof(BStruct), 'wrong total alignment')
        i = 2
        for fname, ftype, fbitsize, fqual in tp.enumfields():
            if fbitsize >= 0:
                continue
            check(layout[i], ffi.offsetof(BStruct, fname), 'wrong offset for field %r' % (fname,))
            if layout[i + 1] != 0:
                BField = ffi._get_cached_btype(ftype)
                check(layout[i + 1], ffi.sizeof(BField), 'wrong size for field %r' % (fname,))
            i += 2
        assert i == len(layout)