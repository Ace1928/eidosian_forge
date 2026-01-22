from collections import deque
from numba.core import types, cgutils
def _do_load(self, builder, ptr, formal_list=None):
    res = []
    for i, i_formal in enumerate(self._pack_map):
        elem_ptr = cgutils.gep_inbounds(builder, ptr, 0, i)
        val = self._models[i_formal].load_from_data_pointer(builder, elem_ptr)
        if formal_list is None:
            res.append((self._fe_types[i_formal], val))
        else:
            formal_list[i_formal] = val
    return res