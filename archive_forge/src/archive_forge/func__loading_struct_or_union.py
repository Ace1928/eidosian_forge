import sys, os
import types
from . import model
from .error import VerificationError
def _loading_struct_or_union(self, tp, prefix, name, module):
    if tp.fldnames is None:
        return
    layoutfuncname = '_cffi_layout_%s_%s' % (prefix, name)
    BFunc = self.ffi._typeof_locked('intptr_t(*)(intptr_t)')[0]
    function = module.load_function(BFunc, layoutfuncname)
    layout = []
    num = 0
    while True:
        x = function(num)
        if x < 0:
            break
        layout.append(x)
        num += 1
    if isinstance(tp, model.StructOrUnion) and tp.partial:
        totalsize = layout[0]
        totalalignment = layout[1]
        fieldofs = layout[2::2]
        fieldsize = layout[3::2]
        tp.force_flatten()
        assert len(fieldofs) == len(fieldsize) == len(tp.fldnames)
        tp.fixedlayout = (fieldofs, fieldsize, totalsize, totalalignment)
    else:
        cname = ('%s %s' % (prefix, name)).strip()
        self._struct_pending_verification[tp] = (layout, cname)