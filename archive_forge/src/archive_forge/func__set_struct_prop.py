from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def _set_struct_prop(self, pname, ptype, pstruct, hints, keys, onerror):
    if isinstance(hints, rq.DictWrapper):
        keys.update(hints._data)
    else:
        keys.update(hints)
    value = pstruct.to_binary(*(), **keys)
    self.change_property(pname, ptype, 32, value, onerror=onerror)